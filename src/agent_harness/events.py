"""Events: what happened during a run, as structured data anyone can listen to (Phase 5).

Observation only (design.md D7): observers receive events after the fact and can
never change what the agent does -- that is what hooks are for. An observer that
raises or is too slow is logged and skipped; the run carries on.

The vocabulary is open on purpose. `Event.type` is a dotted string, not an enum,
and `Event.data` is a free-form dict, so a later phase (or an application, or a
tool) emits a new kind of event without changing this module:

    await emit_event("approval.requested", {"tool": "restart_pod"})   # from anywhere inside a run

`emit_event` finds the current run through a context variable, so code deep
inside the run (the retry wrapper, a tool body, a future subagent) needs no
emitter passed to it; outside a run it does nothing.

Built-in event types (see EventType) and the names reserved for later phases are
listed in docs/design.md section 11.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol, TextIO, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_harness.messages import new_id, short_id, utcnow

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class EventType:
    """Names of the events the harness itself emits. Plain strings: anything else is allowed too."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    CONTEXT_BUILT = "context.built"
    LLM_STARTED = "llm.started"
    LLM_COMPLETED = "llm.completed"
    LLM_FAILED = "llm.failed"
    LLM_RETRYING = "llm.retrying"
    LLM_TRUNCATED = "llm.truncated"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOLS_PARALLEL_GROUP = "tools.parallel_group"
    TOOL_RESULT_OFFLOADED = "tool.result_offloaded"
    COMPACTION_STARTED = "compaction.started"
    COMPACTION_COMPLETED = "compaction.completed"
    COMPACTION_SKIPPED = "compaction.skipped"
    COMPACTION_FAILED = "compaction.failed"
    GUARD_REPEATED_CALL = "guard.repeated_call"
    TOOL_AWAITING_INPUT = "tool.awaiting_input"
    TOOL_DEFERRED = "tool.deferred"
    INTERACTION_REQUESTED = "interaction.requested"
    INTERACTION_RESOLVED = "interaction.resolved"
    CHECKPOINT_SAVED = "checkpoint.saved"
    # Streaming (Phase 8). All transient: shown live, never saved.
    LLM_TOKEN = "llm.token"
    LLM_TOOL_CALL_DELTA = "llm.tool_call_delta"
    LLM_STREAM_RESET = "llm.stream_reset"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("evt_"))
    type: str
    # 1, 2, 3... within one run, in emit order: sort by this, not by timestamp.
    seq: int
    timestamp: datetime = Field(default_factory=utcnow)
    execution_id: str
    agent_id: str
    # Set for a run started by another run (subagents, Phase 11).
    parent_execution_id: str | None = None
    step: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    # Short-lived events (e.g. streamed text chunks, Phase 8) that persistence should skip.
    transient: bool = False
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        """One-line JSON. Values JSON can't represent are written with str()."""
        return json.dumps(self.model_dump(mode="python"), default=str, ensure_ascii=False)


@runtime_checkable
class Observer(Protocol):
    async def on_event(self, event: Event) -> None: ...


# An Observer object, or a plain function (sync or async) taking the event.
ObserverLike = Union[Observer, Callable[[Event], Union[None, Awaitable[None]]]]


class Subscription:
    """One observer plus the event types it wants. `unsubscribe()` stops delivery."""

    def __init__(self, observer: ObserverLike, types: Iterable[str] | None, owner: EventBus | None) -> None:
        self.observer = observer
        # fnmatch patterns: "tool.*", "approval.*", "*". None = everything.
        self.types = tuple(types) if types is not None else None
        self._owner = owner

    def wants(self, event_type: str) -> bool:
        return self.types is None or any(fnmatch.fnmatchcase(event_type, p) for p in self.types)

    def unsubscribe(self) -> None:
        if self._owner is not None:
            self._owner._remove(self)
            self._owner = None

    @property
    def name(self) -> str:
        target = self.observer
        return getattr(target, "__qualname__", None) or type(target).__name__


class EventBus:
    """Observers for every run on a `Runtime`. Delivery is in order and awaited per observer.

    `observer_timeout` bounds each delivery; a slow consumer (a websocket, say) should hand
    events to its own queue rather than do slow work inside `on_event`.
    """

    def __init__(self, *, observer_timeout: float | None = 5.0) -> None:
        self.observer_timeout = observer_timeout
        self._subscriptions: list[Subscription] = []

    def subscribe(self, observer: ObserverLike, *, types: Iterable[str] | None = None) -> Subscription:
        subscription = Subscription(observer, types, self)
        self._subscriptions.append(subscription)
        return subscription

    def _remove(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    @property
    def subscriptions(self) -> list[Subscription]:
        return list(self._subscriptions)


async def deliver(subscription: Subscription, event: Event, timeout: float | None) -> None:
    """Hand one event to one observer. Never raises (except cancellation of the run itself)."""
    if not subscription.wants(event.type):
        return
    observer = subscription.observer
    handler = observer.on_event if isinstance(observer, Observer) else observer
    try:
        outcome = handler(event)
        if inspect.isawaitable(outcome):
            if timeout is None:
                await outcome
            else:
                async with asyncio.timeout(timeout):
                    await outcome
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning("event observer %s took longer than %.1fs on %s; skipped", subscription.name, timeout,
                       event.type)
    except Exception:
        logger.warning("event observer %s failed on %s; ignored", subscription.name, event.type, exc_info=True)


class RunEmitter:
    """Stamps events for one run (ids, step, seq) and delivers them to the bus and the run's own observers.

    `step` is read from `state` (anything with a `.step`) at emit time, so events
    always carry the run's current step without callers passing it.
    """

    def __init__(
        self,
        *,
        execution_id: str,
        agent_id: str,
        bus: EventBus | None = None,
        observers: Iterable[ObserverLike | Subscription] = (),
        state: Any = None,
        parent_execution_id: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.agent_id = agent_id
        self.parent_execution_id = parent_execution_id
        self._bus = bus
        self._timeout = bus.observer_timeout if bus is not None else 5.0
        self._own = [o if isinstance(o, Subscription) else Subscription(o, None, None) for o in observers]
        self._state = state
        # Continue numbering when a run is resumed (e.g. after waiting for input): seq is per
        # execution, not per run() call. Kept in state so it survives save/load.
        self._seq = self._counter().get("last_seq", 0)
        # Parallel tool tasks emit concurrently; the lock keeps every observer seeing seq order.
        self._lock = asyncio.Lock()

    def _counter(self) -> dict[str, Any]:
        extensions = getattr(self._state, "extensions", None)
        return extensions.setdefault("events", {}) if isinstance(extensions, dict) else {}

    async def emit(self, type: str, data: dict[str, Any] | None = None, /, *, transient: bool = False) -> Event:
        self._seq += 1
        self._counter()["last_seq"] = self._seq
        event = Event(
            type=type, seq=self._seq, execution_id=self.execution_id, agent_id=self.agent_id,
            parent_execution_id=self.parent_execution_id,
            step=getattr(self._state, "step", None) if self._state is not None else None,
            data=data or {}, transient=transient,
        )
        subscriptions = (self._bus.subscriptions if self._bus is not None else []) + self._own
        if not subscriptions:
            return event
        if _delivering.get():
            # An observer emitting from inside on_event: the lock is already held by this same
            # delivery, so deliver directly instead of deadlocking on it.
            await self._deliver_all(subscriptions, event)
        else:
            async with self._lock:
                await self._deliver_all(subscriptions, event)
        return event

    async def _deliver_all(self, subscriptions: list[Subscription], event: Event) -> None:
        token = _delivering.set(True)
        try:
            for subscription in subscriptions:
                await deliver(subscription, event, self._timeout)
        finally:
            _delivering.reset(token)


_current: ContextVar[RunEmitter | None] = ContextVar("agent_harness_run_emitter", default=None)
_delivering: ContextVar[bool] = ContextVar("agent_harness_event_delivering", default=False)


def current_emitter() -> RunEmitter | None:
    """The emitter of the run this code is executing in, or None outside a run."""
    return _current.get()


def bind_emitter(emitter: RunEmitter | None):
    """Make `emitter` current for this task (and tasks it starts). Returns a token for `unbind_emitter`."""
    return _current.set(emitter)


def unbind_emitter(token) -> None:
    _current.reset(token)


async def emit_event(type: str, data: dict[str, Any] | None = None, /, *, transient: bool = False) -> Event | None:
    """Emit an event for the current run. A no-op (returns None) outside a run."""
    emitter = _current.get()
    if emitter is None:
        return None
    return await emitter.emit(type, data, transient=transient)


# --- ready-made observers -------------------------------------------------------


class JsonlEventWriter:
    """Appends every event as one JSON line to `path`: a complete, replayable record of runs."""

    def __init__(self, path: str | Path, *, include_transient: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_transient = include_transient

    async def on_event(self, event: Event) -> None:
        if event.transient and not self.include_transient:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")


class PrintObserver:
    """One compact line per event, for watching a run live in the terminal."""

    def __init__(self, stream: TextIO | None = None, *, max_value_chars: int = 80,
                 skip: Iterable[str] = ("llm.token",)) -> None:
        self.stream = stream
        self.max_value_chars = max_value_chars
        # One line per streamed piece would flood the terminal; pass skip=() to see them.
        self.skip = tuple(skip)

    async def on_event(self, event: Event) -> None:
        if any(fnmatch.fnmatchcase(event.type, pattern) for pattern in self.skip):
            return
        fields = " ".join(f"{k}={self._short(v)}" for k, v in event.data.items())
        step = f" step={event.step}" if event.step else ""
        print(f"[event] exec={short_id(event.execution_id)} #{event.seq}{step} {event.type} {fields}".rstrip(),
              file=self.stream or sys.stderr, flush=True)

    def _short(self, value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        text = " ".join(text.split())
        return text if len(text) <= self.max_value_chars else text[: self.max_value_chars] + "…"
