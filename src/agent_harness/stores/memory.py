"""`InMemoryStore`: a `PersistenceStore` kept in a dict.

For scripts, checks and single-process use. Everything is copied in and out as JSON-shaped
data, so it behaves like a database: changing a loaded state does not change what is stored.
It is also the reference for what a store must do; `PostgresStore` follows it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from agent_harness.errors import ErrorCode, PersistenceConflictError, ValidationError
from agent_harness.events import Event
from agent_harness.messages import Message, new_id, utcnow
from agent_harness.persistence import (
    DEFAULT_LEASE_SECONDS,
    ChatPage,
    ChatRecord,
    Lease,
    RunRecord,
    StoredMessage,
    check_claim,
    check_history,
    decode_cursor,
    derive_title,
    encode_cursor,
    event_row,
    restore_state,
    state_snapshot,
)
from agent_harness.state import AgentState, RunStatus


@dataclass
class _Chat:
    record: ChatRecord
    state: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)   # {seq, run_id, message}
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: dict[int, dict[str, Any]] = field(default_factory=dict)  # seq -> event row
    lease_token: str | None = None


class InMemoryStore:
    def __init__(self, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._chats: dict[str, _Chat] = {}
        self._lock = asyncio.Lock()
        # Injectable so a check can move time forward (lease expiry) without sleeping.
        self._clock = clock

    # --- used by the runtime --------------------------------------------------------

    async def acquire(
        self, chat_id: str, *, agent_id: str, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Lease:
        async with self._lock:
            now = self._clock()
            chat = self._chats.get(chat_id)
            created = chat is None
            if chat is None:
                chat = _Chat(ChatRecord(id=chat_id, agent_id=agent_id, owner_id=owner_id, status=RunStatus.PENDING,
                                        metadata=dict(metadata or {}), created_at=now, updated_at=now))
                self._chats[chat_id] = chat
            record = chat.record
            check_claim(chat_id, stored_agent_id=record.agent_id, stored_owner_id=record.owner_id,
                        deleted_at=record.deleted_at, lease_expires_at=record.lease_expires_at,
                        agent_id=agent_id, owner_id=owner_id, now=now)
            chat.lease_token = new_id("lease_")
            record.owner_id = record.owner_id or owner_id
            record.lease_expires_at = now + timedelta(seconds=lease_seconds)
            record.version += 1
            return Lease(
                chat_id=chat_id, token=chat.lease_token, lease_seconds=lease_seconds, created=created,
                recovered=record.status is RunStatus.RUNNING,
                state=self._state(chat), message_count=len(chat.messages),
                last_message_id=chat.messages[-1]["message"]["id"] if chat.messages else None,
            )

    async def save(
        self, lease: Lease, state: AgentState, run: RunRecord, *, events: Sequence[Event] = (), release: bool = False,
    ) -> int:
        async with self._lock:
            chat = self._leased(lease)
            check_history(len(chat.messages), chat.messages[-1]["message"]["id"] if chat.messages else None, state)
            start = len(chat.messages)
            for seq, message in enumerate(state.messages[start:], start=start):
                chat.messages.append({"seq": seq, "run_id": run.id, "message": message.model_dump(mode="json")})
            chat.runs[run.id] = run.model_dump(mode="json")
            for event in events:
                chat.events.setdefault(event.seq, {**event_row(event), "run_id": run.id})
            chat.state = state_snapshot(state)
            now, record = self._clock(), chat.record
            record.status = state.status
            record.message_count = len(chat.messages)
            record.title = record.title or derive_title(state.messages)
            record.input_tokens, record.output_tokens = state.usage.input_tokens, state.usage.output_tokens
            record.cost_usd = state.usage.cost_usd
            record.version += 1
            record.updated_at = now
            if len(chat.messages) > start:
                record.last_message_at = now
            if release:
                chat.lease_token, record.lease_expires_at = None, None
            else:
                record.lease_expires_at = now + timedelta(seconds=lease.lease_seconds)
            return record.version

    async def renew(self, lease: Lease) -> None:
        async with self._lock:
            self._leased(lease).record.lease_expires_at = self._clock() + timedelta(seconds=lease.lease_seconds)

    async def release(self, lease: Lease) -> None:
        async with self._lock:
            chat = self._chats.get(lease.chat_id)
            if chat is None or chat.lease_token != lease.token:
                return
            if lease.created and not chat.messages and not chat.state:
                del self._chats[lease.chat_id]  # created by this lease and never saved
                return
            chat.lease_token, chat.record.lease_expires_at = None, None

    # --- reading ----------------------------------------------------------------------

    async def get_chat(self, chat_id: str) -> ChatRecord | None:
        chat = self._chats.get(chat_id)
        return chat.record.model_copy(deep=True) if chat else None

    async def list_chats(
        self, *, owner_id: str | None = None, status: RunStatus | str | None = None,
        metadata: dict[str, Any] | None = None, include_archived: bool = False, limit: int = 50,
        cursor: str | None = None,
    ) -> ChatPage:
        after = decode_cursor(cursor) if cursor else None
        rows = [
            c.record for c in self._chats.values()
            if c.record.deleted_at is None
            and (owner_id is None or c.record.owner_id == owner_id)
            and (status is None or c.record.status == RunStatus(status))
            and (include_archived or c.record.archived_at is None)
            and all(c.record.metadata.get(k) == v for k, v in (metadata or {}).items())
            and (after is None or (c.record.updated_at, c.record.id) < after)
        ]
        rows.sort(key=lambda r: (r.updated_at, r.id), reverse=True)
        page = [r.model_copy(deep=True) for r in rows[:limit]]
        return ChatPage(items=page, next_cursor=encode_cursor(page[-1]) if len(rows) > limit else None)

    async def load_state(self, chat_id: str) -> AgentState | None:
        chat = self._chats.get(chat_id)
        return self._state(chat) if chat else None

    async def get_messages(
        self, chat_id: str, *, before_seq: int | None = None, limit: int | None = None,
    ) -> list[StoredMessage]:
        chat = self._chats.get(chat_id)
        if chat is None:
            return []
        rows = [m for m in chat.messages if before_seq is None or m["seq"] < before_seq]
        if limit is not None:
            rows = rows[-limit:] if limit else []
        return [StoredMessage(chat_id=chat_id, seq=m["seq"], run_id=m["run_id"],
                              message=Message.model_validate(m["message"])) for m in rows]

    async def list_runs(self, chat_id: str) -> list[RunRecord]:
        chat = self._chats.get(chat_id)
        runs = [RunRecord.model_validate(r) for r in chat.runs.values()] if chat else []
        return sorted(runs, key=lambda r: r.started_at)

    async def get_events(self, chat_id: str, *, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        chat = self._chats.get(chat_id)
        if chat is None:
            return []
        rows = [chat.events[s] for s in sorted(chat.events) if s > after_seq]
        rows = rows[:limit] if limit is not None else rows
        return [Event.model_validate({k: v for k, v in r.items() if k != "run_id"}) for r in rows]

    # --- managing chats -------------------------------------------------------------

    async def create_chat(
        self, state: AgentState, *, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        title: str | None = None, forked_from: tuple[str, int] | None = None,
    ) -> ChatRecord:
        async with self._lock:
            if state.execution_id in self._chats:
                raise PersistenceConflictError(f"chat '{state.execution_id}' already exists")
            now = self._clock()
            record = ChatRecord(
                id=state.execution_id, agent_id=state.agent_id, owner_id=owner_id,
                title=title or derive_title(state.messages), status=state.status, metadata=dict(metadata or {}),
                message_count=len(state.messages), input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens, cost_usd=state.usage.cost_usd, version=1,
                forked_from_id=forked_from[0] if forked_from else None,
                forked_from_seq=forked_from[1] if forked_from else None,
                created_at=now, updated_at=now, last_message_at=now if state.messages else None,
            )
            self._chats[record.id] = _Chat(
                record, state=state_snapshot(state),
                messages=[{"seq": i, "run_id": None, "message": m.model_dump(mode="json")}
                          for i, m in enumerate(state.messages)],
            )
            return record.model_copy(deep=True)

    async def update_chat(
        self, chat_id: str, *, title: str | None = None, metadata: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> ChatRecord:
        async with self._lock:
            chat = self._chats.get(chat_id)
            if chat is None or chat.record.deleted_at is not None:
                raise ValidationError(f"no chat '{chat_id}'", code=ErrorCode.INVALID_CONFIG)
            record = chat.record
            if title is not None:
                record.title = title
            if metadata:
                record.metadata = {**record.metadata, **metadata}
            if archived is not None:
                record.archived_at = (record.archived_at or self._clock()) if archived else None
            record.version += 1
            return record.model_copy(deep=True)

    async def delete_chat(self, chat_id: str, *, hard: bool = False) -> bool:
        async with self._lock:
            chat = self._chats.get(chat_id)
            if chat is None:
                return False
            if hard:
                del self._chats[chat_id]
            else:
                chat.record.deleted_at = chat.record.deleted_at or self._clock()
                chat.record.version += 1
            return True

    async def close(self) -> None:
        return None

    # --- internals --------------------------------------------------------------------

    def _leased(self, lease: Lease) -> _Chat:
        chat = self._chats.get(lease.chat_id)
        if chat is None or chat.lease_token != lease.token:
            raise PersistenceConflictError(
                f"lost the lease on chat '{lease.chat_id}' (it expired and another run took the chat over)"
            )
        return chat

    @staticmethod
    def _state(chat: _Chat) -> AgentState | None:
        if not chat.state:
            return None
        return restore_state(chat.state, [m["message"] for m in chat.messages])
