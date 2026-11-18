"""`ChatService`: what the Chat API does, with no web framework in sight.

Chats live in the runtime's `PersistenceStore` (Phase 9); this class starts runs on them, keeps the
runs that are active in this process (so clients can watch them), and applies the ownership rules.
`server.py` only translates HTTP/WebSocket to calls on it, so everything here can be driven from a
script.

Ownership: `owner` is whatever the caller is allowed to see (a user id, an incident number, ...).
`None` means no restriction. A chat with an owner is invisible (not found) to a different owner.

One run per chat at a time. This process refuses a second message while a run is active here; the
store's lease refuses it across processes. Active runs are in this process's memory, so with several
server processes a client should reach the same one for a chat (sticky routing); a client that lands
elsewhere just reads the saved chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from agent_harness.agent import Agent
from agent_harness.api.broadcast import RunBroadcast
from agent_harness.api.models import (
    AnswerIn,
    ChatDetail,
    ChatList,
    ChatOut,
    ChatUpdate,
    MessageOut,
)
from agent_harness.errors import ErrorCode, ErrorInfo, HarnessError, ValidationError
from agent_harness.hitl import Answer, pending_interactions
from agent_harness.messages import short_id
from agent_harness.persistence import ChatRecord, PersistenceStore
from agent_harness.runtime import Execution, Runtime
from agent_harness.state import AgentState, RunStatus

logger = logging.getLogger(__name__)

DEFAULT_PAGE = 100
MAX_PAGE = 500


class NotFoundError(HarnessError):
    """No such chat, or it belongs to someone else (the two are indistinguishable on purpose)."""

    default_code = "NOT_FOUND"


class BusyError(HarnessError):
    """The chat already has a run, or this server is at its limit of active runs."""

    default_code = ErrorCode.PERSISTENCE_CONFLICT


class TooManyRunsError(HarnessError):
    default_code = "TOO_MANY_RUNS"


@dataclass
class _ActiveRun:
    execution: Execution
    broadcast: RunBroadcast
    watcher: asyncio.Task[None]


class ChatService:
    def __init__(
        self, runtime: Runtime, agent: Agent, *, max_active_runs: int | None = 50, keep_finished: float = 60.0,
        max_buffered_pieces: int = 5000,
    ) -> None:
        if runtime.store is None:
            raise ValueError("the Chat API needs a runtime with a PersistenceStore: chats are loaded and saved there")
        self.runtime = runtime
        self.agent = agent
        self.store: PersistenceStore = runtime.store
        self.max_active_runs = max_active_runs
        # How long a finished run stays watchable (a late or reconnecting client still sees its end).
        self.keep_finished = keep_finished
        self.max_buffered_pieces = max_buffered_pieces
        self._runs: dict[str, _ActiveRun] = {}
        self._retire: set[asyncio.Task[None]] = set()

    # --- chats ---------------------------------------------------------------------------

    @property
    def active_runs(self) -> int:
        return sum(1 for r in self._runs.values() if not r.broadcast.finished)

    def _is_active(self, chat_id: str) -> bool:
        run = self._runs.get(chat_id)
        return run is not None and not run.broadcast.finished

    def _out(self, record: ChatRecord) -> ChatOut:
        return ChatOut.from_record(record, active=self._is_active(record.id))

    async def _owned(self, chat_id: str, owner: str | None) -> ChatRecord:
        record = await self.store.get_chat(chat_id)
        if (record is None or record.deleted_at is not None
                or (owner is not None and record.owner_id is not None and record.owner_id != owner)):
            raise NotFoundError(f"no chat '{chat_id}'")
        return record

    async def create_chat(self, owner: str | None, *, title: str | None = None,
                          metadata: dict[str, Any] | None = None) -> ChatOut:
        record = await self.store.create_chat(AgentState(agent_id=self.agent.id), owner_id=owner,
                                              metadata=metadata, title=title)
        return self._out(record)

    async def list_chats(
        self, owner: str | None, *, status: str | None = None, include_archived: bool = False,
        limit: int = 50, cursor: str | None = None,
    ) -> ChatList:
        page = await self.store.list_chats(owner_id=owner, status=status, include_archived=include_archived,
                                           limit=min(max(limit, 1), 200), cursor=cursor)
        return ChatList(items=[self._out(r) for r in page.items], next_cursor=page.next_cursor)

    async def get_chat(self, chat_id: str, owner: str | None, *, limit: int = DEFAULT_PAGE,
                       before_seq: int | None = None) -> ChatDetail:
        record = await self._owned(chat_id, owner)
        limit = min(max(limit, 1), MAX_PAGE)
        stored = await self.store.get_messages(chat_id, before_seq=before_seq, limit=limit)
        pending = []
        if record.status is RunStatus.AWAITING_INPUT:
            state = await self.store.load_state(chat_id)
            pending = pending_interactions(state) if state is not None else []
        return ChatDetail(
            chat=self._out(record), messages=[MessageOut.from_stored(m) for m in stored],
            has_more=bool(stored) and stored[0].seq > 0, pending_interactions=pending,
            streaming=self._is_active(chat_id),
        )

    async def update_chat(self, chat_id: str, owner: str | None, change: ChatUpdate) -> ChatOut:
        await self._owned(chat_id, owner)
        record = await self.store.update_chat(chat_id, title=change.title, metadata=change.metadata,
                                              archived=change.archived)
        return self._out(record)

    async def delete_chat(self, chat_id: str, owner: str | None) -> None:
        await self._owned(chat_id, owner)
        if self._is_active(chat_id):
            raise BusyError(f"chat '{chat_id}' is running; cancel it first")
        await self.store.delete_chat(chat_id)

    # --- runs ----------------------------------------------------------------------------

    async def send_message(self, chat_id: str, owner: str | None, text: str) -> None:
        """Start a run on the chat with the person's message. Returns once the run has started
        (the chat is leased and the message saved); watch it with `stream`."""
        record = await self._owned(chat_id, owner)
        if record.status is RunStatus.AWAITING_INPUT:
            raise BusyError(f"chat '{chat_id}' is waiting for answers; send them with POST .../answers first")
        await self._start(chat_id, owner, text, answers=[])

    async def answer(self, chat_id: str, owner: str | None, answers: Sequence[AnswerIn]) -> None:
        """Resume a run that is waiting for a person, with their answers."""
        record = await self._owned(chat_id, owner)
        if record.status is not RunStatus.AWAITING_INPUT:
            raise BusyError(f"chat '{chat_id}' is '{record.status.value}', not waiting for answers")
        converted = [Answer(interaction_id=a.interaction_id, action=a.action, arguments=a.arguments, value=a.value,
                            note=a.note, answered_by=owner) for a in answers]
        await self._start(chat_id, owner, None, answers=converted)

    async def _start(self, chat_id: str, owner: str | None, text: str | None, *, answers: list[Answer]) -> None:
        if self._is_active(chat_id):
            raise BusyError(f"chat '{chat_id}' is already running; wait for it to finish or cancel it")
        if self.max_active_runs is not None and self.active_runs >= self.max_active_runs:
            raise TooManyRunsError(f"this server is already running {self.active_runs} chats; try again shortly")
        broadcast = RunBroadcast(chat_id, max_transient=self.max_buffered_pieces)
        # `owner` is checked above; passing it on makes the store enforce it too.
        execution = self.runtime.start(self.agent, text, chat_id=chat_id, owner_id=owner, observers=[broadcast],
                                       answers=answers)
        watcher = asyncio.ensure_future(self._watch(chat_id, execution, broadcast))
        self._runs[chat_id] = _ActiveRun(execution, broadcast, watcher)
        # run.started is the first event, emitted once the chat is leased and the message saved. If the
        # run ends first, it never started (chat busy in another process, invalid answers, ...): report why.
        starting = asyncio.ensure_future(broadcast.started.wait())
        try:
            await asyncio.wait({starting, watcher}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            starting.cancel()
        if broadcast.start_failure is not None:
            self._runs.pop(chat_id, None)
            failure = broadcast.start_failure
            if isinstance(failure, HarnessError) and failure.code == ErrorCode.PERSISTENCE_CONFLICT:
                raise BusyError(failure.message, details=failure.details) from failure
            raise failure

    async def _watch(self, chat_id: str, execution: Execution, broadcast: RunBroadcast) -> None:
        try:
            result = await execution.wait()
        except asyncio.CancelledError:
            broadcast.finish(RunStatus.CANCELLED)
            raise
        except BaseException as exc:
            broadcast.start_failure = exc
            if not isinstance(exc, HarnessError):
                logger.exception("chat=%s run failed to start", short_id(chat_id))
            broadcast.finish(RunStatus.FAILED, ErrorInfo.from_exception(exc))
        else:
            broadcast.finish(result.status, result.error or result.persistence_error)
        self._schedule_retire(chat_id, broadcast)

    def _schedule_retire(self, chat_id: str, broadcast: RunBroadcast) -> None:
        async def retire() -> None:
            await asyncio.sleep(self.keep_finished)
            run = self._runs.get(chat_id)
            if run is not None and run.broadcast is broadcast:
                del self._runs[chat_id]

        task = asyncio.ensure_future(retire())
        self._retire.add(task)
        task.add_done_callback(self._retire.discard)

    async def cancel(self, chat_id: str, owner: str | None, reason: str | None = None) -> bool:
        """Stop the chat's active run. False if none is running in this process."""
        await self._owned(chat_id, owner)
        run = self._runs.get(chat_id)
        if run is None or run.broadcast.finished:
            return False
        run.execution.cancel(reason or "cancelled by the client")
        return True

    async def watch(self, chat_id: str, owner: str | None) -> RunBroadcast | None:
        """The chat's run in this process (running, or finished a moment ago), to `subscribe()` to.
        None if nothing is running here: read the saved chat instead."""
        await self._owned(chat_id, owner)
        run = self._runs.get(chat_id)
        return run.broadcast if run is not None else None

    async def status(self, chat_id: str, owner: str | None) -> RunStatus:
        return (await self._owned(chat_id, owner)).status

    async def shutdown(self, *, grace: float = 5.0) -> None:
        """Stop active runs (each saves itself as cancelled) and clean up."""
        runs = [r for r in self._runs.values() if not r.broadcast.finished]
        for run in runs:
            run.execution.cancel("server shutting down")
        if runs:
            await asyncio.wait([r.watcher for r in runs], timeout=grace)
        for task in list(self._retire):
            task.cancel()
        for task in list(self._retire):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._runs.clear()
