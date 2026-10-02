"""Provider-neutral message vocabulary.

Messages are immutable and made of typed parts, so tool calls (and later
images or reasoning) sit alongside text without schema changes. Adapters
translate these to and from provider wire formats.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def short_id(id: str) -> str:
    """Compact form of an id for log lines."""
    return id[:8]


def preview(text: str, limit: int = 200) -> str:
    """Single-line, length-capped text for log lines."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}… (+{len(flat) - limit} chars)"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A model's request to invoke a tool.

    `arguments` holds the parsed object when the provider supplied valid JSON.
    `raw_arguments` keeps the original string so malformed arguments surface
    as a tool error instead of being lost.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None

    @classmethod
    def parse(cls, id: str, name: str, raw_arguments: str | None) -> ToolCall:
        arguments: dict[str, Any] | None = None
        if raw_arguments is not None:
            try:
                value = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                arguments = value
        return cls(id=id, name=name, arguments=arguments, raw_arguments=raw_arguments)


class TextPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    call: ToolCall


Part = Annotated[TextPart | ToolCallPart, Field(discriminator="type")]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    role: Role
    parts: tuple[Part, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check_role_invariants(self) -> Message:
        if self.role is Role.TOOL:
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
        elif self.tool_call_id is not None or self.is_error:
            raise ValueError("only tool messages may set tool_call_id or is_error")
        if self.role is not Role.ASSISTANT and any(isinstance(p, ToolCallPart) for p in self.parts):
            raise ValueError("only assistant messages may contain tool calls")
        return self

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p.call for p in self.parts if isinstance(p, ToolCallPart)]

    @classmethod
    def system(cls, text: str, **kwargs: Any) -> Message:
        return cls(role=Role.SYSTEM, parts=(TextPart(text=text),), **kwargs)

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> Message:
        return cls(role=Role.USER, parts=(TextPart(text=text),), **kwargs)

    @classmethod
    def assistant(cls, text: str | None = None, tool_calls: Sequence[ToolCall] = (), **kwargs: Any) -> Message:
        parts: list[TextPart | ToolCallPart] = [TextPart(text=text)] if text else []
        parts.extend(ToolCallPart(call=c) for c in tool_calls)
        return cls(role=Role.ASSISTANT, parts=tuple(parts), **kwargs)

    @classmethod
    def tool(
        cls, tool_call_id: str, content: str, *, name: str | None = None, is_error: bool = False, **kwargs: Any
    ) -> Message:
        return cls(
            role=Role.TOOL,
            parts=(TextPart(text=content),),
            tool_call_id=tool_call_id,
            name=name,
            is_error=is_error,
            **kwargs,
        )
