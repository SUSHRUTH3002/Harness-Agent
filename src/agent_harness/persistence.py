"""Persistence (Phase 9): runs save themselves, chats load by id, one run per chat at a time.

A *chat* is one `AgentState` (its `execution_id` is the chat id) plus what a chat UI needs
around it: an owner key (a user id, an incident number, ...), a title, timestamps, archive and
delete flags, the messages, one record per run, and the event log. A `PersistenceStore` keeps
all of it; the runtime decides when to save:

- when a run starts (the person's message is stored before the model is called),
- after every step that leads to another step (a checkpoint),
- when the run ends, whatever its status (completed, awaiting_input, failed, cancelled, ...).

Only one run works on a chat at a time. `acquire` takes a *lease* (a token with an expiry,
renewed while the run is alive); a second request for the same chat is refused with
`PersistenceConflictError` before anything runs. If a process dies mid-run its lease expires
and the next run on that chat recovers it: tool calls left without a result are closed as
interrupted, not re-run, because they may already have had effects.

Messages are append-only (state.py), so a save writes only the messages added since the last
one; the rest of the state is stored as a snapshot next to them.

Stores: `InMemoryStore` (stores/memory.py, for scripts and checks) and `PostgresStore`
(stores/postgres.py). Another database works by implementing `PersistenceStore`.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from agent_harness.compaction import EXTENSION_KEY as COMPACTION_KEY
from agent_harness.errors import (
    ErrorCode,
    ErrorInfo,
    PersistenceConflictError,
    PersistenceError,
    ValidationError,
)
from agent_harness.events import Event, EventType, emit_event
from agent_harness.messages import Message, Role, new_id, short_id, utcnow
from agent_harness.state import AgentState, RunStatus

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 600.0
TITLE_MAX_CHARS = 80


# --- what a chat UI reads ---------------------------------------------------------


class ChatRecord(BaseModel):
    """One chat as a chat list shows it (no messages)."""

    id: str
    agent_id: str
    # Whatever the app groups chats by: a user id, an incident number, a team, ...
    owner_id: str | None = None
    title: str | None = None
    status: RunStatus
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    # Bumped on every write; lets an app detect that its copy is out of date.
    version: int = 0
    # Set when this chat was forked from another (edit a message, regenerate an answer).
    forked_from_id: str | None = None
    forked_from_seq: int | None = None
    created_at: datetime
    # Last time the conversation changed (a run saved). Renaming or archiving does not move it,
    # so a chat list ordered by it stays ordered by activity.
    updated_at: datetime
    last_message_at: datetime | None = None
    archived_at: datetime | None = None
    deleted_at: datetime | None = None
    lease_expires_at: datetime | None = None

    @property
    def running(self) -> bool:
        """A run is working on this chat right now."""
        return self.lease_expires_at is not None and self.lease_expires_at > utcnow()

    @property
    def interrupted(self) -> bool:
        """The last run stopped without finishing (its process ended); the next run recovers it."""
        return self.status is RunStatus.RUNNING and not self.running


class ChatPage(BaseModel):
    items: list[ChatRecord]
    # Pass to list_chats(cursor=...) for the next page; None when there is none.
    next_cursor: str | None = None


class StoredMessage(BaseModel):
    chat_id: str
    # Position in the conversation, from 0.
    seq: int
    run_id: str | None = None
    message: Message


class RunRecord(BaseModel):
    """One run(), or resume(), on a chat: which messages it added and what it cost."""

    id: str = Field(default_factory=lambda: new_id("run_"))
    chat_id: str
    kind: Literal["new", "continue", "resume"]
    status: RunStatus = RunStatus.RUNNING
    # The previous run on this chat ended without saving (its process died) and was recovered.
    recovered: bool = False
    # The messages this run added: first_seq..last_seq.
    first_seq: int
    last_seq: int | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    error: ErrorInfo | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


@dataclass
class Lease:
    """The right to run a chat, from `acquire` until the final `save(release=True)` or `release`."""

    chat_id: str
    token: str
    lease_seconds: float
    # This acquire created the chat.
    created: bool
    # The previous run ended without saving its end; the runtime recovers the state.
    recovered: bool
    # The stored state, or None for a new chat.
    state: AgentState | None
    message_count: int
    last_message_id: str | None


# --- the store contract -------------------------------------------------------------


@runtime_checkable
class PersistenceStore(Protocol):
    """Where chats live. The runtime uses the first four methods; a chat UI uses the rest."""

    async def acquire(
        self, chat_id: str, *, agent_id: str, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Lease:
        """Lease the chat for one run, creating it if new. Raises PersistenceConflictError if
        another run holds it, ValidationError if it was deleted or belongs to another agent/owner."""
        ...

    async def save(
        self, lease: Lease, state: AgentState, run: RunRecord, *, events: Sequence[Event] = (), release: bool = False,
    ) -> int:
        """Write new messages, the state snapshot, the run record and events, in one transaction.
        Returns the chat's new version. Raises PersistenceConflictError if the lease was lost."""
        ...

    async def renew(self, lease: Lease) -> None:
        """Extend the lease (the runtime calls this while a run is alive)."""
        ...

    async def release(self, lease: Lease) -> None:
        """Give the lease up without a final save (a chat created by it and never saved is removed)."""
        ...

    async def get_chat(self, chat_id: str) -> ChatRecord | None: ...

    async def list_chats(
        self, *, owner_id: str | None = None, status: RunStatus | str | None = None,
        metadata: dict[str, Any] | None = None, include_archived: bool = False, limit: int = 50,
        cursor: str | None = None,
    ) -> ChatPage:
        """Most recently active first. `metadata` matches chats containing those top-level keys/values."""
        ...

    async def load_state(self, chat_id: str) -> AgentState | None: ...

    async def get_messages(
        self, chat_id: str, *, before_seq: int | None = None, limit: int | None = None,
    ) -> list[StoredMessage]:
        """In conversation order. With `limit`, the latest `limit` messages before `before_seq`."""
        ...

    async def list_runs(self, chat_id: str) -> list[RunRecord]: ...

    async def get_events(self, chat_id: str, *, after_seq: int = 0, limit: int | None = None) -> list[Event]: ...

    async def create_chat(
        self, state: AgentState, *, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        title: str | None = None, forked_from: tuple[str, int] | None = None,
    ) -> ChatRecord:
        """Store a chat that was built outside a run (a fork, an import)."""
        ...

    async def update_chat(
        self, chat_id: str, *, title: str | None = None, metadata: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> ChatRecord:
        """Rename, archive/unarchive, or merge keys into metadata. None leaves a field unchanged."""
        ...

    async def delete_chat(self, chat_id: str, *, hard: bool = False) -> bool:
        """Soft delete (hidden, still in the database) unless `hard`. Returns False if not found."""
        ...

    async def close(self) -> None: ...


# --- helpers shared by store implementations ---------------------------------------


def state_snapshot(state: AgentState) -> dict[str, Any]:
    """Everything in the state except the messages, which are stored as rows."""
    return state.model_dump(mode="json", exclude={"messages"})


def restore_state(snapshot: dict[str, Any], messages: Sequence[dict[str, Any]]) -> AgentState:
    return AgentState.model_validate({**snapshot, "messages": list(messages)})


def derive_title(messages: Sequence[Message]) -> str | None:
    """A first title from the person's first message (an app can rename the chat later)."""
    for message in messages:
        if message.role is Role.USER and message.text.strip():
            text = " ".join(message.text.split())
            return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1] + "…"
    return None


def check_claim(
    chat_id: str, *, stored_agent_id: str, stored_owner_id: str | None, deleted_at: datetime | None,
    lease_expires_at: datetime | None, agent_id: str, owner_id: str | None, now: datetime,
) -> None:
    """Can a new run lease this chat? Raises if not."""
    if deleted_at is not None:
        raise ValidationError(f"chat '{chat_id}' was deleted", code=ErrorCode.INVALID_CONFIG)
    if stored_agent_id != agent_id:
        raise ValidationError(f"chat '{chat_id}' belongs to agent '{stored_agent_id}', not '{agent_id}'",
                              code=ErrorCode.INVALID_CONFIG)
    if owner_id is not None and stored_owner_id is not None and stored_owner_id != owner_id:
        raise ValidationError(f"chat '{chat_id}' belongs to another owner", code=ErrorCode.INVALID_CONFIG)
    if lease_expires_at is not None and lease_expires_at > now:
        raise PersistenceConflictError(
            f"chat '{chat_id}' is already running (another request is working on it); try again when it finishes",
            details={"lease_expires_at": lease_expires_at.isoformat()},
        )


def check_history(stored_count: int, stored_last_id: str | None, state: AgentState) -> None:
    """The stored messages must be the start of `state.messages`: history only grows."""
    if stored_count > len(state.messages) or (
        stored_count and state.messages[stored_count - 1].id != stored_last_id
    ):
        raise PersistenceConflictError(
            f"chat '{state.execution_id}' has changed in the store since this state was loaded; load it again"
        )


def encode_cursor(record: ChatRecord) -> str:
    raw = f"{record.updated_at.isoformat()}|{record.id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        stamp, chat_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        return datetime.fromisoformat(stamp), chat_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValidationError("invalid cursor", code=ErrorCode.INVALID_CONFIG) from exc


def event_row(event: Event) -> dict[str, Any]:
    """An event as JSON-safe data (values JSON can't hold are written with str(), as in to_json)."""
    return json.loads(event.to_json())


# --- forking: how a chat UI edits a message or regenerates an answer ----------------


def fork_state(state: AgentState, before_seq: int, new_chat_id: str) -> AgentState:
    """A new chat holding the first `before_seq` messages of `state`.

    It must end right before one of the person's messages (or at the end of a finished
    conversation), so it never contains a tool call without its result.
    """
    count = len(state.messages)
    if not 1 <= before_seq <= count:
        raise ValidationError(f"before_seq must be between 1 and {count}", code=ErrorCode.INVALID_CONFIG)
    if before_seq < count and state.messages[before_seq].role is not Role.USER:
        raise ValidationError("a chat can only be forked right before one of the user's messages",
                              code=ErrorCode.INVALID_CONFIG)
    messages = state.messages[:before_seq]
    call_ids = {c.id for m in messages for c in m.tool_calls}
    extensions = copy.deepcopy(state.extensions)
    extensions.pop("hitl", None)    # nothing is waiting at a fork point
    extensions.pop("events", None)  # a new chat numbers its events from 1
    compaction = extensions.get(COMPACTION_KEY)
    if compaction and (compaction.get("covered_until") or 0) > before_seq:
        extensions.pop(COMPACTION_KEY)  # the summary covers messages the fork does not have
    forked = AgentState(
        execution_id=new_chat_id, agent_id=state.agent_id, messages=list(messages), step=state.step,
        tool_results=[r for r in state.tool_results if r.call_id in call_ids],
        metadata=copy.deepcopy(state.metadata), extensions=extensions,
    )
    if forked.pending_tool_calls():
        raise ValidationError("the fork point leaves a tool call without its result", code=ErrorCode.INVALID_CONFIG)
    return forked


async def fork_chat(
    store: PersistenceStore, chat_id: str, *, before_seq: int, new_chat_id: str | None = None,
    owner_id: str | None = None, title: str | None = None,
) -> ChatRecord:
    """Start a new chat from the first `before_seq` messages of another.

    Edit a message: fork before it, then run the new chat with the edited text.
    Regenerate an answer: fork before the user message that produced it, then send it again.
    The original chat is left as it was; the fork records where it came from.
    """
    record = await store.get_chat(chat_id)
    state = await store.load_state(chat_id)
    if record is None or state is None or record.deleted_at is not None:
        raise ValidationError(f"no chat '{chat_id}'", code=ErrorCode.INVALID_CONFIG)
    forked = fork_state(state, before_seq, new_chat_id or new_id())
    return await store.create_chat(
        forked, owner_id=owner_id if owner_id is not None else record.owner_id, metadata=record.metadata,
        title=title or record.title, forked_from=(chat_id, before_seq),
    )


# --- what the runtime uses during a run -----------------------------------------------


class RunPersistence:
    """Saves one run: right after it starts, after each step that continues, and when it ends.

    It is also an observer on the run's events, collecting them so each save writes them in
    the same transaction as the state. A failed checkpoint is logged and the run carries on
    (the next save writes everything since); a lost lease stops the run, because someone
    else now owns the chat. A failed final save is returned, not raised.
    """

    def __init__(self, store: PersistenceStore, lease: Lease, run: RunRecord, state: AgentState) -> None:
        self.store = store
        self.lease = lease
        self.run = run
        self._usage = state.usage.model_copy()
        self._steps = state.step
        self._events: list[Event] = []
        self._heartbeat: asyncio.Task[None] | None = None

    async def on_event(self, event: Event) -> None:
        if not event.transient:
            self._events.append(event)

    def _update_run(self, state: AgentState, final: bool) -> None:
        run, usage = self.run, state.usage
        run.status = state.status
        run.last_seq = len(state.messages) - 1 if len(state.messages) > run.first_seq else None
        run.steps = state.step - self._steps
        run.input_tokens = usage.input_tokens - self._usage.input_tokens
        run.output_tokens = usage.output_tokens - self._usage.output_tokens
        if usage.cost_usd is not None:
            run.cost_usd = usage.cost_usd - (self._usage.cost_usd or 0.0)
        run.error = state.error
        if final:
            run.finished_at = state.finished_at or utcnow()

    async def save(self, state: AgentState, *, final: bool = False) -> int:
        self._update_run(state, final)
        events = list(self._events)
        version = await self.store.save(self.lease, state, self.run, events=events, release=final)
        del self._events[: len(events)]
        logger.debug("exec=%s saved: version=%d status=%s messages=%d events=%d%s", short_id(self.lease.chat_id),
                     version, state.status.value, len(state.messages), len(events), " (final)" if final else "")
        await emit_event(EventType.CHECKPOINT_SAVED, {
            "version": version, "status": state.status.value, "messages": len(state.messages),
            "events": len(events), "final": final,
        }, transient=True)
        return version

    async def checkpoint(self, state: AgentState) -> None:
        """Called by the loop after a step that leads to another one."""
        try:
            await self.save(state)
        except PersistenceConflictError:
            raise
        except Exception as exc:
            logger.warning("exec=%s step=%d checkpoint not saved (the next save will catch up): %s",
                           short_id(self.lease.chat_id), state.step, exc)

    async def finish(self, state: AgentState) -> ErrorInfo | None:
        """Save the final state and release the chat. Returns the error if that failed."""
        self.stop_heartbeat()
        try:
            await self.save(state, final=True)
            return None
        except asyncio.CancelledError:
            error = ErrorInfo(code=ErrorCode.PERSISTENCE_ERROR, message="cancelled while saving the final state",
                              type="CancelledError")
        except PersistenceError as exc:
            error = exc.to_info()
        except Exception as exc:
            error = ErrorInfo(code=ErrorCode.PERSISTENCE_ERROR, message=f"{type(exc).__name__}: {exc}",
                              type=type(exc).__name__)
        logger.error("exec=%s final state NOT saved: [%s] %s", short_id(self.lease.chat_id), error.code, error.message)
        try:
            await self.store.release(self.lease)
        except Exception:
            logger.debug("lease release failed too", exc_info=True)
        return error

    def start_heartbeat(self) -> None:
        self._heartbeat = asyncio.ensure_future(self._beat())

    def stop_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            self._heartbeat = None

    async def _beat(self) -> None:
        interval = max(self.lease.lease_seconds / 3, 0.05)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.store.renew(self.lease)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("exec=%s could not renew the chat lease: %s", short_id(self.lease.chat_id), exc)
