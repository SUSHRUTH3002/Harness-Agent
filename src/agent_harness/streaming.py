"""Streaming support (Phase 8): turn a stream of pieces into the same `LLMResponse` as before.

    provider.stream(request) ──▶ text / tool_call / usage / finish pieces ──▶ StreamAssembler ──▶ LLMResponse

Only the finished response matters to the rest of the harness (tools, persistence, HITL).
The pieces exist so a person can watch the answer being written; the loop announces them as
transient events (`llm.token`, ...) that are never saved.

- `open_stream(llm, request)` streams from any provider: one that cannot stream is called with
  `generate()` and its whole response replayed as pieces (so callers need no special case).
- `timed_stream(...)` adds an idle timeout (longest gap between pieces) and a total timeout.
- `StreamAssembler` joins the pieces. Tool-call arguments are joined as text and parsed only
  once the stream is finished, so half-written JSON is never used.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import AsyncIterator

from agent_harness.errors import ErrorCode, LLMError
from agent_harness.llm import (
    FinishReason,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    StreamingLLMProvider,
    Usage,
)
from agent_harness.messages import Message, ToolCall, new_id


def response_events(response: LLMResponse) -> list[LLMStreamEvent]:
    """A whole response as pieces: its text, each tool call, usage, and a finish carrying the response."""
    events: list[LLMStreamEvent] = []
    if response.message.text:
        events.append(LLMStreamEvent(kind="text", text=response.message.text))
    for index, call in enumerate(response.message.tool_calls):
        events.append(LLMStreamEvent(kind="tool_call", index=index, call_id=call.id, name=call.name))
    if response.usage_reported:
        events.append(LLMStreamEvent(kind="usage", usage=response.usage))
    events.append(LLMStreamEvent(kind="finish", finish_reason=response.finish_reason, model=response.model,
                                 response=response))
    return events


async def _replay(llm: LLMProvider, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
    response = await llm.generate(request)
    for event in response_events(response):
        yield event


def open_stream(llm: LLMProvider, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
    """Stream from `llm`, or, if it cannot stream, call `generate()` and replay the whole response."""
    if isinstance(llm, StreamingLLMProvider):
        return llm.stream(request)
    return _replay(llm, request)


async def close_iterator(iterator: object) -> None:
    """Close an async iterator (and with it the provider's connection), ignoring ones that can't be closed."""
    close = getattr(iterator, "aclose", None) or getattr(iterator, "close", None)
    if close is None:
        return
    outcome = close()
    if inspect.isawaitable(outcome):
        await outcome


async def timed_stream(
    events: AsyncIterator[LLMStreamEvent], *, total: float | None = None, idle: float | None = None
) -> AsyncIterator[LLMStreamEvent]:
    """Pass pieces through, failing with a retryable `LLMError(TIMEOUT)` if the stream stalls.

    `idle` bounds the gap between two pieces, counted from the first piece: waiting for the first
    one is bounded by `total` (and the provider's own request timeout) instead, because a model
    can think for a long time before it writes anything. `total` bounds the whole stream.
    The source is always closed when this ends, so a stalled connection is dropped.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total if total is not None else None
    started = False
    iterator = aiter(events)
    try:
        while True:
            limit, cause = None, None
            if deadline is not None:
                limit, cause = max(deadline - loop.time(), 0.0), "total"
            if idle is not None and started and (limit is None or idle < limit):
                limit, cause = idle, "idle"
            try:
                if limit is None:
                    event = await anext(iterator)
                else:
                    async with asyncio.timeout(limit):
                        event = await anext(iterator)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                message = (f"the model sent nothing for {idle:.0f}s while streaming" if cause == "idle"
                           else f"LLM call timed out after {total:.0f}s")
                raise LLMError(message, code=ErrorCode.TIMEOUT, retryable=True,
                               details={"stream_timeout": cause}) from exc
            started = True
            yield event
    finally:
        await close_iterator(iterator)


class _CallSlot:
    __slots__ = ("call_id", "name", "arguments")

    def __init__(self) -> None:
        self.call_id: str | None = None
        self.name: str | None = None
        self.arguments: list[str] = []


class StreamAssembler:
    """Collects stream pieces into one `LLMResponse`.

    Tool-call pieces are grouped by `index`. Some providers number every call 0, so a piece that
    names a *different* call id than the one being built starts a new call instead of merging.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._text: list[str] = []
        self._calls: list[_CallSlot] = []
        self._by_index: dict[int, _CallSlot] = {}
        self._usage: Usage | None = None
        self._finish: FinishReason | None = None
        self._model: str | None = None
        self._response: LLMResponse | None = None
        self.text_chars = 0

    @property
    def finished(self) -> bool:
        return self._finish is not None

    def add(self, event: LLMStreamEvent) -> tuple[str | None, str] | None:
        """Take one piece. When it names a tool call for the first time, returns that call's
        (id, name) so it can be announced ("preparing get_metrics"); otherwise None."""
        if event.kind == "reset":
            self.reset()
        elif event.kind == "text":
            self._text.append(event.text)
            self.text_chars += len(event.text)
        elif event.kind == "tool_call":
            slot = self._by_index.get(event.index)
            if slot is None or (event.call_id and slot.call_id and event.call_id != slot.call_id):
                slot = _CallSlot()
                self._calls.append(slot)
                self._by_index[event.index] = slot
            if event.call_id and not slot.call_id:
                slot.call_id = event.call_id
            named = False
            if event.name and not slot.name:
                slot.name = event.name
                named = True
            if event.arguments:
                slot.arguments.append(event.arguments)
            if named:
                return slot.call_id, slot.name
        elif event.kind == "usage":
            self._usage = event.usage
        elif event.kind == "finish":
            self._finish = event.finish_reason or FinishReason.STOP
            self._model = event.model or self._model
            self._response = event.response
        return None

    @property
    def text(self) -> str:
        return "".join(self._text)

    def response(self) -> LLMResponse:
        """The finished response. Raises `LLMError` if the stream never finished or was empty."""
        if self._response is not None:
            return self._response
        if self._finish is None:
            raise LLMError("the response stream ended before the model finished", code=ErrorCode.TRANSPORT,
                           retryable=True)
        text = self.text
        calls = [
            ToolCall.parse(slot.call_id or new_id("call_"), slot.name or "",
                           "".join(slot.arguments) if slot.arguments else "{}")
            for slot in self._calls
        ]
        if self._finish is FinishReason.LENGTH:
            finish = FinishReason.LENGTH
        elif calls:
            finish = FinishReason.TOOL_CALLS
        else:
            finish = FinishReason.STOP
        if not text and not calls and finish is not FinishReason.LENGTH:
            raise LLMError("provider returned an empty response", code=ErrorCode.EMPTY_RESPONSE, retryable=True)
        return LLMResponse(
            message=Message.assistant(text, tool_calls=calls, metadata={"source": "model", "model": self._model}),
            finish_reason=finish,
            usage=self._usage or Usage(),
            model=self._model,
            usage_reported=self._usage is not None,
        )


__all__ = ["StreamAssembler", "close_iterator", "open_stream", "response_events", "timed_stream"]
