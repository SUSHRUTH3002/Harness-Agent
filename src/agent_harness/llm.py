"""Provider-independent LLM interface.

The core never imports a provider SDK. Adapters implement `LLMProvider`,
translate `LLMRequest` to their wire format, and raise `LLMError` with a
neutral code on failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from agent_harness.messages import Message, Role


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    # The model hit its output-token limit; the message is cut off.
    LENGTH = "length"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Only set when the provider reports it; the harness keeps no price table.
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=cost,
        )


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSchema] = Field(default_factory=list)
    model: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    message: Message
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    # False when the provider sent no token counts (some stream without them, e.g. many local
    # servers): `usage` is then zero, not an estimate, and nothing should calibrate from it.
    usage_reported: bool = True
    raw: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_assistant(self) -> LLMResponse:
        if self.message.role is not Role.ASSISTANT:
            raise ValueError("LLMResponse.message must be an assistant message")
        return self


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class LLMStreamEvent(BaseModel):
    """One piece of a streamed model response, in a provider-neutral form (Phase 8).

    kind:
      text       `text` is the next piece of the answer.
      tool_call  a piece of tool call number `index`: `call_id`/`name` when first known, then
                 `arguments` fragments to append. Arguments are only parsed once complete.
      usage      token counts for the whole response (usually last; absent if not reported).
      finish     the model is done: `finish_reason`. `response` is set when the whole response
                 was already available (a provider that does not stream), and is then used as is.
      reset      the attempt so far is discarded (a retry started): forget what came before.
    """

    kind: Literal["text", "tool_call", "usage", "finish", "reset"]
    text: str = ""
    index: int = 0
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    model: str | None = None
    response: LLMResponse | None = None
    attempt: int | None = None


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    """An `LLMProvider` that can also stream (Phase 8). `generate()` stays the non-streaming path.

    `stream()` yields `LLMStreamEvent`s and ends with a `finish` event; it raises `LLMError`
    (neutral codes, as `generate()` does) if the call fails, including partway through.
    Closing the iterator early must close the connection to the model.
    """

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]: ...
