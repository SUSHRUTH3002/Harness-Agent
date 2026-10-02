"""LiteLLM adapter: one `LLMProvider` for every provider LiteLLM supports.

Switch providers by changing the model string, e.g. "openai/<model>",
"anthropic/<model>", "gemini/<model>", "deepseek/<model>", "ollama/<model>".
Credentials come from the provider's usual environment variables
(OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) or from `api_key` / `api_base`.

Requires the optional extra:  pip install 'agent-harness[litellm]'
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Awaitable, Callable

from agent_harness.errors import ErrorCode, LLMError
from agent_harness.llm import FinishReason, LLMRequest, LLMResponse, ToolSchema, Usage
from agent_harness.messages import Message, Role, ToolCall, new_id

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]
_UNLOGGED_PARAMS = {"api_key", "messages", "tools", "model"}

# Matched against the exception's class hierarchy by name, most specific first,
# so the mapping survives LiteLLM version changes without importing its types.
_ERROR_MAP: list[tuple[str, str, bool]] = [
    ("ContextWindowExceededError", ErrorCode.CONTEXT_WINDOW_EXCEEDED, False),
    ("ContentPolicyViolationError", ErrorCode.INVALID_REQUEST, False),
    ("AuthenticationError", ErrorCode.AUTH, False),
    ("PermissionDeniedError", ErrorCode.AUTH, False),
    ("RateLimitError", ErrorCode.RATE_LIMIT, True),
    ("Timeout", ErrorCode.TIMEOUT, True),
    ("APITimeoutError", ErrorCode.TIMEOUT, True),
    ("ServiceUnavailableError", ErrorCode.SERVER, True),
    ("InternalServerError", ErrorCode.SERVER, True),
    ("APIConnectionError", ErrorCode.TRANSPORT, True),
    ("NotFoundError", ErrorCode.INVALID_REQUEST, False),
    ("BadRequestError", ErrorCode.INVALID_REQUEST, False),
]
_MAX_ERROR_MESSAGE = 1000


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a LiteLLM object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_litellm_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages -> OpenAI-style chat messages (LiteLLM's input format)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role is Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": c.raw_arguments
                            if c.raw_arguments is not None
                            else json.dumps(c.arguments or {}),
                        },
                    }
                    for c in m.tool_calls
                ]
            out.append(entry)
        elif m.role is Role.TOOL:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
        else:
            out.append({"role": m.role.value, "content": m.text})
    return out


def to_litellm_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


def map_error(exc: BaseException) -> LLMError:
    names = {cls.__name__ for cls in type(exc).__mro__}
    code, retryable = ErrorCode.LLM_ERROR, False
    for name, mapped_code, mapped_retryable in _ERROR_MAP:
        if name in names:
            code, retryable = mapped_code, mapped_retryable
            break
    else:
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            code = ErrorCode.AUTH
        elif status == 429:
            code, retryable = ErrorCode.RATE_LIMIT, True
        elif status == 408:
            code, retryable = ErrorCode.TIMEOUT, True
        elif isinstance(status, int) and status >= 500:
            code, retryable = ErrorCode.SERVER, True
        elif isinstance(status, int) and 400 <= status < 500:
            code = ErrorCode.INVALID_REQUEST
    message = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_MESSAGE]
    details = {"provider_error": type(exc).__name__}
    if getattr(exc, "status_code", None) is not None:
        details["status_code"] = exc.status_code
    return LLMError(message, code=code, retryable=retryable, details=details)


def _usage(response: Any, cost: float | None) -> Usage:
    usage = _get(response, "usage")
    prompt_details = _get(usage, "prompt_tokens_details")
    cache_read = _get(prompt_details, "cached_tokens") or _get(usage, "cache_read_input_tokens") or 0
    cache_write = _get(usage, "cache_creation_input_tokens") or 0
    return Usage(
        input_tokens=_get(usage, "prompt_tokens") or 0,
        output_tokens=_get(usage, "completion_tokens") or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=cost,
    )


def from_litellm_response(response: Any, cost: float | None = None) -> LLMResponse:
    choices = _get(response, "choices") or []
    if not choices:
        raise LLMError("provider returned no choices", code=ErrorCode.EMPTY_RESPONSE, retryable=True)
    choice = choices[0]
    raw_message = _get(choice, "message")
    text = _get(raw_message, "content") or ""
    calls = []
    for tc in _get(raw_message, "tool_calls") or []:
        function = _get(tc, "function")
        raw_args = _get(function, "arguments")
        if isinstance(raw_args, dict):  # a few providers return parsed arguments
            raw_args = json.dumps(raw_args)
        calls.append(ToolCall.parse(_get(tc, "id") or new_id("call_"), _get(function, "name") or "", raw_args))

    finish = _get(choice, "finish_reason")
    if finish == "length":
        finish_reason = FinishReason.LENGTH
    elif calls:
        finish_reason = FinishReason.TOOL_CALLS
    else:
        finish_reason = FinishReason.STOP
    if not text and not calls and finish_reason is not FinishReason.LENGTH:
        raise LLMError("provider returned an empty response", code=ErrorCode.EMPTY_RESPONSE, retryable=True)

    model = _get(response, "model")
    return LLMResponse(
        message=Message.assistant(text, tool_calls=calls, metadata={"source": "model", "model": model}),
        finish_reason=finish_reason,
        usage=_usage(response, cost),
        model=model,
        raw=response,
    )


class LiteLLMProvider:
    """`LLMProvider` backed by `litellm.acompletion`.

    `model` is the default; `Agent.model` overrides it per request. Extra keyword
    arguments (temperature, max_tokens, ...) become defaults for every call, and
    `Agent.model_settings` overrides them. `completion_fn` exists for tests.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        drop_params: bool = True,
        completion_fn: CompletionFn | None = None,
        **default_params: Any,
    ) -> None:
        if completion_fn is None:
            try:
                import litellm
            except ImportError as exc:
                raise ImportError(
                    "LiteLLMProvider requires the 'litellm' extra: pip install 'agent-harness[litellm]'"
                ) from exc
            completion_fn = litellm.acompletion
        self.model = model
        self._completion = completion_fn
        self._params: dict[str, Any] = dict(default_params)
        # drop_params lets one agent config work across providers that don't support every setting.
        self._params["drop_params"] = drop_params
        for key, value in (("api_key", api_key), ("api_base", api_base), ("timeout", timeout)):
            if value is not None:
                self._params[key] = value

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.model
        if not model:
            raise LLMError("no model configured: set Agent.model or LiteLLMProvider(model=...)",
                           code=ErrorCode.INVALID_REQUEST)
        kwargs: dict[str, Any] = {**self._params, **request.settings}
        kwargs["model"] = model
        kwargs["messages"] = to_litellm_messages(request.messages)
        if request.tools:
            kwargs["tools"] = to_litellm_tools(request.tools)
        # Never log api_key: only the non-secret settings are listed.
        settings = {k: v for k, v in kwargs.items() if k not in _UNLOGGED_PARAMS}
        logger.debug("litellm request: model=%s messages=%d tools=%d settings=%s",
                     model, len(kwargs["messages"]), len(kwargs.get("tools", ())), settings)
        started = time.perf_counter()

        try:
            response = await self._completion(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            error = map_error(exc)
            logger.warning("litellm call to %s failed after %.2fs: [%s] retryable=%s %s", model,
                           time.perf_counter() - started, error.code, error.retryable, error.details)
            raise error from exc
        result = from_litellm_response(response, cost=self._cost(response))
        logger.debug(
            "litellm response from %s in %.2fs: finish=%s tokens in=%d out=%d cost=%s",
            result.model or model, time.perf_counter() - started, result.finish_reason.value,
            result.usage.input_tokens, result.usage.output_tokens,
            f"${result.usage.cost_usd:.6f}" if result.usage.cost_usd is not None else "n/a",
        )
        return result

    @staticmethod
    def _cost(response: Any) -> float | None:
        hidden = getattr(response, "_hidden_params", None)
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        return float(cost) if isinstance(cost, (int, float)) else None
