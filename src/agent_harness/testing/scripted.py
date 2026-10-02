"""`ScriptedProvider`: an `LLMProvider` that replays a script instead of calling a model."""

from __future__ import annotations

import inspect
import json
import logging
from collections import deque
from typing import Any, Awaitable, Callable, Iterable, Union

from agent_harness.errors import LLMError
from agent_harness.llm import FinishReason, LLMRequest, LLMResponse, Usage
from agent_harness.messages import Message, ToolCall, new_id

logger = logging.getLogger(__name__)

Responder = Callable[[LLMRequest], Union[LLMResponse, Awaitable[LLMResponse]]]
ScriptItem = Union[LLMResponse, str, BaseException, Responder]


def call(name: str, arguments: dict[str, Any] | None = None, *, id: str | None = None) -> ToolCall:
    """Build a well-formed tool call, as a provider adapter would."""
    raw = json.dumps(arguments or {})
    return ToolCall(id=id or new_id("call_"), name=name, arguments=arguments or {}, raw_arguments=raw)


def text_response(text: str, *, usage: Usage | None = None) -> LLMResponse:
    return LLMResponse(message=Message.assistant(text), finish_reason=FinishReason.STOP, usage=usage or Usage())


def tool_call_response(*calls: ToolCall, text: str | None = None, usage: Usage | None = None) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant(text, tool_calls=calls), finish_reason=FinishReason.TOOL_CALLS, usage=usage or Usage()
    )


def truncated_response(text: str = "", *, tool_calls: Iterable[ToolCall] = ()) -> LLMResponse:
    """A response cut off at the output-token limit (finish_reason=length)."""
    return LLMResponse(message=Message.assistant(text, tool_calls=tuple(tool_calls)), finish_reason=FinishReason.LENGTH)


class ScriptedProvider:
    """Returns scripted items in order; records every request it receives.

    Each item may be an `LLMResponse`, a `str` (a final text answer), an
    exception instance (raised), or a callable taking the `LLMRequest` and
    returning a response (sync or async). `fallback` is used once the script
    runs out; without one, running out raises `LLMError`.
    """

    def __init__(self, script: Iterable[ScriptItem] = (), *, fallback: ScriptItem | None = None) -> None:
        self._script: deque[ScriptItem] = deque(script)
        self._fallback = fallback
        self.requests: list[LLMRequest] = []

    def add(self, *items: ScriptItem) -> ScriptedProvider:
        self._script.extend(items)
        return self

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._script:
            item = self._script.popleft()
        elif self._fallback is not None:
            item = self._fallback
        else:
            logger.warning("scripted provider exhausted after %d request(s)", len(self.requests) - 1)
            raise LLMError("ScriptedProvider script is exhausted", code="SCRIPT_EXHAUSTED")
        logger.debug("scripted response #%d: %s (%d left)", len(self.requests), type(item).__name__, len(self._script))

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return text_response(item)
        if isinstance(item, LLMResponse):
            return item
        result = item(request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return text_response(result)
        return result
