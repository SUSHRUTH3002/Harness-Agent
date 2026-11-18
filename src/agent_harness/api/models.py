"""Request and response shapes of the Chat API (Phase 11a). Only pydantic, no web framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from agent_harness.hitl import Interaction
from agent_harness.messages import Role
from agent_harness.persistence import ChatRecord, StoredMessage
from agent_harness.state import RunStatus

MAX_MESSAGE_CHARS = 100_000


class ChatCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatUpdate(BaseModel):
    """Only the fields you send change."""

    title: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] | None = None
    archived: bool | None = None


class MessageCreate(BaseModel):
    text: str = Field(max_length=MAX_MESSAGE_CHARS)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class AnswerIn(BaseModel):
    """One person's response to one open request (see `Interaction`)."""

    interaction_id: str
    action: Literal["approve", "edit", "reject", "answer"]
    # edit: the replacement arguments.
    arguments: dict[str, Any] | None = None
    # answer: the picked option label(s) or the person's own text.
    value: Any = None
    note: str | None = Field(default=None, max_length=10_000)


class AnswersIn(BaseModel):
    answers: list[AnswerIn] = Field(min_length=1)


class CancelIn(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class ChatOut(BaseModel):
    id: str
    title: str | None
    owner_id: str | None
    status: RunStatus
    # A run is working on it now (this process, or another one holding its lease).
    running: bool
    # The last run stopped without finishing (its process ended); the next message recovers it.
    interrupted: bool
    metadata: dict[str, Any]
    message_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    # Changes on every write: lets a client tell that its copy is out of date.
    version: int
    forked_from_id: str | None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None
    archived_at: datetime | None

    @classmethod
    def from_record(cls, record: ChatRecord, *, active: bool = False) -> ChatOut:
        return cls(
            id=record.id, title=record.title, owner_id=record.owner_id, status=record.status,
            running=record.running or active, interrupted=record.interrupted, metadata=record.metadata,
            message_count=record.message_count, input_tokens=record.input_tokens,
            output_tokens=record.output_tokens, cost_usd=record.cost_usd, version=record.version,
            forked_from_id=record.forked_from_id, created_at=record.created_at, updated_at=record.updated_at,
            last_message_at=record.last_message_at, archived_at=record.archived_at,
        )


class ChatList(BaseModel):
    items: list[ChatOut]
    # Pass as ?cursor= for the next page; null when there is none.
    next_cursor: str | None = None


class ToolCallOut(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] | None


class MessageOut(BaseModel):
    # Position in the conversation, from 0.
    seq: int
    id: str
    role: Role
    text: str
    # assistant: the tools it asked for. tool: the call this message answers.
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    created_at: datetime

    @classmethod
    def from_stored(cls, stored: StoredMessage) -> MessageOut:
        m = stored.message
        return cls(
            seq=stored.seq, id=m.id, role=m.role, text=m.text,
            tool_calls=[ToolCallOut(id=c.id, name=c.name, arguments=c.arguments) for c in m.tool_calls],
            tool_call_id=m.tool_call_id, name=m.name, is_error=m.is_error, created_at=m.created_at,
        )


class ChatDetail(BaseModel):
    chat: ChatOut
    # The latest messages (or those before `before_seq`), oldest first.
    messages: list[MessageOut]
    # Older messages exist: request them with ?before_seq=<first seq shown>.
    has_more: bool
    # When the run is waiting for a person: what it asks (answer with POST .../answers).
    pending_interactions: list[Interaction] = Field(default_factory=list)
    # A run is streaming in this process: connect to .../stream to watch it.
    streaming: bool = False


class RunStarted(BaseModel):
    chat_id: str
    status: Literal["running"] = "running"
    # Watch the run here (WebSocket).
    stream: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    active_runs: int
