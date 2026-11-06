"""Run events: a typed record of what happened during a run, delivered as it happens.

A minimal first slice of Phase 5. Pass `on_event=` to `Runtime.start()/run()`
and it is called with an `Event` for each of:

    run.started       {input}
    step.started      {}
    llm.response      {text, tool_calls, finish_reason, usage, duration}
    tool.started      {call_id, name, arguments}
    tool.completed    {call_id, name, is_error, error_code, content, chars, duration}
    tool.offloaded    {call_id, name, artifact_id, tokens}
    context.reduced   {groups_dropped, estimated_tokens}       (Phase 3 dropping)
    context.compacted {reason, messages, region_tokens, summary_tokens, archive}
    run.finished      {status, output, error, usage, steps, duration}

Events are observational: a sink that raises is logged and ignored, never
allowed to break the run. With no sink, nothing is built or sent.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from agent_harness.messages import short_id, utcnow

logger = logging.getLogger(__name__)

# Tool output can be huge; events carry at most this much of it (the transcript keeps all of it).
EVENT_CONTENT_CHARS = 20_000


class Event(BaseModel):
    type: str
    execution_id: str
    agent_id: str
    step: int
    data: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)


EventSink = Callable[[Event], Awaitable[None] | None]


class EventEmitter:
    """Binds a sink to one run, so call sites only say what happened."""

    def __init__(self, sink: EventSink | None = None, *, execution_id: str = "", agent_id: str = "") -> None:
        self.sink = sink
        self.execution_id = execution_id
        self.agent_id = agent_id

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    async def emit(self, type: str, step: int, /, **data: Any) -> None:
        if self.sink is None:
            return
        event = Event(type=type, execution_id=self.execution_id, agent_id=self.agent_id, step=step, data=data)
        try:
            outcome = self.sink(event)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:
            logger.warning("exec=%s event sink failed on %s; ignoring", short_id(self.execution_id), type,
                           exc_info=True)


def clip(text: str, limit: int = EVENT_CONTENT_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}\n…({len(text) - limit} more characters)"
