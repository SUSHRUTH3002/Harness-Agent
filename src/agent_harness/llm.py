"""Provider-independent LLM interface.

The core never imports a provider SDK. Adapters implement `LLMProvider`,
translate `LLMRequest` to their wire format, and raise `LLMError` with a
neutral code on failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

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
    raw: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_assistant(self) -> LLMResponse:
        if self.message.role is not Role.ASSISTANT:
            raise ValueError("LLMResponse.message must be an assistant message")
        return self


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
