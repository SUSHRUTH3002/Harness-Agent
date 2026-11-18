"""Local OpenAI-compatible model adapter (Ollama, vLLM, llama.cpp, LM Studio, ...).

Talks directly to the server via the `openai` SDK, wrapped with `instructor.from_openai(...,
mode=instructor.Mode.JSON)`. Every call passes `response_model=None` explicitly (instructor
1.17's `AsyncInstructor.create()` requires the parameter, with no default — see
`instructor/v2/core/client.py`), and `OpenAIJSONHandler.prepare_request` returns the rest of
the kwargs completely unchanged whenever `response_model is None` (`instructor/v2/providers/
openai/handlers.py`) — so every call behaves exactly like calling `AsyncOpenAI.chat.completions.
create` directly. The client is built this way only for consistency with other code that
already uses `instructor` against local servers; it does no structured-output extraction here.

The request/response wire format is identical to LiteLLM's (both speak plain OpenAI Chat
Completions), so this module reuses `litellm_provider`'s translation functions and error
mapping rather than duplicating them.

Streams too (Phase 8), with the same chunk parser as LiteLLM. Token counts are asked for with
`stream_options.include_usage`; a server that doesn't send them gives "usage unknown"
(`LLMResponse.usage_reported=False`), never an estimate. If a server rejects that option, pass
`stream_usage=False`.

Requires the optional extra:  pip install 'agent-harness[local]'
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from agent_harness.errors import ErrorCode, LLMError
from agent_harness.llm import LLMRequest, LLMResponse, LLMStreamEvent
from agent_harness.providers.litellm_provider import (
    from_litellm_response as parse_openai_response,
    map_error,
    openai_stream_events,
    to_litellm_messages as to_openai_messages,
    to_litellm_tools as to_openai_tools,
)

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]


def _build_client(base_url: str, api_key: str, mode: Any | None) -> CompletionFn:
    try:
        import instructor
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "LocalProvider requires the 'local' extra: pip install 'agent-harness[local]'"
        ) from exc
    client = instructor.from_openai(AsyncOpenAI(base_url=base_url, api_key=api_key), mode=mode or instructor.Mode.JSON)
    return client.chat.completions.create


class LocalProvider:
    """`LLMProvider` for a local, OpenAI-compatible model server.

    `model` is the plain model id the server reports — there is no "provider/" prefix here,
    unlike `LiteLLMProvider`. `base_url` is required unless `completion_fn` is supplied directly
    (tests do this to avoid building a real client). `api_key` defaults to a placeholder because
    most local servers don't check it, but the OpenAI SDK requires a non-empty string.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str = "not-needed",
        mode: Any | None = None,
        timeout: float | None = None,
        stream_usage: bool = True,
        completion_fn: CompletionFn | None = None,
        **default_params: Any,
    ) -> None:
        if completion_fn is None:
            if not base_url:
                raise LLMError(
                    "LocalProvider requires base_url (set HARNESS_API_BASE)", code=ErrorCode.INVALID_REQUEST
                )
            completion_fn = _build_client(base_url, api_key, mode)
        self.model = model
        self._completion = completion_fn
        self.stream_usage = stream_usage
        self._params: dict[str, Any] = dict(default_params)
        if timeout is not None:
            self._params["timeout"] = timeout

    def _kwargs(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self.model
        if not model:
            raise LLMError(
                "no model configured: set Agent.model or LocalProvider(model=...)", code=ErrorCode.INVALID_REQUEST
            )
        kwargs: dict[str, Any] = {**self._params, **request.settings}
        kwargs["model"] = model
        kwargs["messages"] = to_openai_messages(request.messages)
        if request.tools:
            kwargs["tools"] = to_openai_tools(request.tools)
        # Required positionally by instructor's AsyncInstructor.create(); None makes it a
        # passthrough (see module docstring) rather than structured-output extraction.
        kwargs["response_model"] = None
        return kwargs

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        kwargs = self._kwargs(request)
        model = kwargs["model"]
        kwargs["stream"] = True
        if self.stream_usage:
            kwargs.setdefault("stream_options", {"include_usage": True})
        logger.debug("local model request (stream): model=%s messages=%d tools=%d", model, len(kwargs["messages"]),
                     len(kwargs.get("tools", ())))
        started = time.perf_counter()
        try:
            chunks = await self._completion(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            error = map_error(exc)
            logger.warning("local model stream to %s failed after %.2fs: [%s] retryable=%s %s", model,
                           time.perf_counter() - started, error.code, error.retryable, error.details)
            raise error from exc
        # Local servers don't report cost: no cost function, so usage.cost_usd stays None.
        async with contextlib.aclosing(openai_stream_events(chunks)) as events:
            async for event in events:
                yield event
        logger.debug("local model stream from %s finished in %.2fs", model, time.perf_counter() - started)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._kwargs(request)
        model = kwargs["model"]
        logger.debug("local model request: model=%s messages=%d tools=%d", model, len(kwargs["messages"]),
                     len(kwargs.get("tools", ())))
        started = time.perf_counter()
        try:
            response = await self._completion(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            error = map_error(exc)
            logger.warning("local model call to %s failed after %.2fs: [%s] retryable=%s %s", model,
                           time.perf_counter() - started, error.code, error.retryable, error.details)
            raise error from exc
        # Local servers don't report cost; parse_openai_response leaves usage.cost_usd as None.
        result = parse_openai_response(response)
        logger.debug(
            "local model response from %s in %.2fs: finish=%s tokens in=%d out=%d",
            result.model or model, time.perf_counter() - started, result.finish_reason.value,
            result.usage.input_tokens, result.usage.output_tokens,
        )
        return result
