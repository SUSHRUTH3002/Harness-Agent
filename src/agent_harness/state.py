"""Serializable execution state.

`messages` is append-only history. What happens next is derived from it
(see `pending_tool_calls`) rather than kept in a separate state-machine
variable, so a saved state can always be resumed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.errors import ErrorInfo
from agent_harness.llm import Usage
from agent_harness.messages import Message, Role, ToolCall, new_id
from agent_harness.tools import ToolResult


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"

    @property
    def is_terminal(self) -> bool:
        return self not in (RunStatus.PENDING, RunStatus.RUNNING)


class AgentState(BaseModel):
    execution_id: str = Field(default_factory=new_id)
    agent_id: str
    status: RunStatus = RunStatus.PENDING
    messages: list[Message] = Field(default_factory=list)
    step: int = 0
    usage: Usage = Field(default_factory=Usage)
    tool_results: list[ToolResult] = Field(default_factory=list)
    result: str | None = None
    error: ErrorInfo | None = None
    # Output-truncation continuation bookkeeping (see AgentLoop).
    consecutive_truncations: int = 0
    partial_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Namespaced key/value space for extensions: extensions["<name>"] = ...
    extensions: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def last_assistant_message(self) -> Message | None:
        for message in reversed(self.messages):
            if message.role is Role.ASSISTANT:
                return message
        return None

    def pending_tool_calls(self) -> list[ToolCall]:
        """Tool calls on the last assistant message that have no result yet, in order."""
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].role is Role.ASSISTANT:
                answered = {m.tool_call_id for m in self.messages[index + 1 :] if m.role is Role.TOOL}
                return [c for c in self.messages[index].tool_calls if c.id not in answered]
        return []
