"""`PostgresStore`: chats in PostgreSQL, via asyncpg (`pip install agent-harness[postgres]`).

Tables (prefix `harness_` by default, in `schema`):

    harness_chats      one row per chat: owner, title, status, the state snapshot, lease, timestamps
    harness_messages   one row per message, (chat_id, seq) ordered; append-only
    harness_runs       one row per run()/resume(): message range, status, steps, tokens, error
    harness_events     the run events (Phase 5), (chat_id, seq) ordered
    harness_schema_migrations   which schema versions are applied

The schema is created and upgraded on first use (`migrate=False` to manage it yourself; the
SQL is in `MIGRATIONS`). Every save is one transaction that checks the lease, so two app
servers can share the database safely. Lease expiry uses the database clock.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

import asyncpg

from agent_harness.errors import ErrorCode, HarnessError, PersistenceConflictError, PersistenceError, ValidationError
from agent_harness.events import Event
from agent_harness.messages import Message, new_id
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

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
# pg_advisory_xact_lock key, so two app servers starting together don't both migrate.
_MIGRATION_LOCK = 7_302_118_449_221

# Each entry is applied once, in order, inside a transaction. Never edit an applied entry;
# add a new one. {chats}, {messages}, {runs}, {events} are qualified table names, {p} the prefix.
MIGRATIONS: list[tuple[int, str]] = [
    (1, """
CREATE TABLE IF NOT EXISTS {chats} (
    id               TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL,
    owner_id         TEXT,
    title            TEXT,
    status           TEXT NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    state            JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    message_count    INTEGER NOT NULL DEFAULT 0,
    input_tokens     BIGINT NOT NULL DEFAULT 0,
    output_tokens    BIGINT NOT NULL DEFAULT 0,
    cost_usd         DOUBLE PRECISION,
    version          BIGINT NOT NULL DEFAULT 0,
    lease_token      TEXT,
    lease_expires_at TIMESTAMPTZ,
    forked_from_id   TEXT,
    forked_from_seq  INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at  TIMESTAMPTZ,
    archived_at      TIMESTAMPTZ,
    deleted_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS {p}chats_owner_recent ON {chats} (owner_id, updated_at DESC, id DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS {p}chats_recent ON {chats} (updated_at DESC, id DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS {p}chats_status ON {chats} (status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS {p}chats_metadata ON {chats} USING GIN (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS {p}chats_forked_from ON {chats} (forked_from_id) WHERE forked_from_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {messages} (
    chat_id    TEXT NOT NULL REFERENCES {chats}(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    id         TEXT NOT NULL,
    run_id     TEXT,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL DEFAULT '',
    content    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (chat_id, seq)
);
CREATE INDEX IF NOT EXISTS {p}messages_run ON {messages} (run_id);

CREATE TABLE IF NOT EXISTS {runs} (
    id            TEXT PRIMARY KEY,
    chat_id       TEXT NOT NULL REFERENCES {chats}(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    recovered     BOOLEAN NOT NULL DEFAULT false,
    first_seq     INTEGER NOT NULL,
    last_seq      INTEGER,
    steps         INTEGER NOT NULL DEFAULT 0,
    input_tokens  BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd      DOUBLE PRECISION,
    error         JSONB,
    started_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS {p}runs_chat ON {runs} (chat_id, started_at);

CREATE TABLE IF NOT EXISTS {events} (
    chat_id    TEXT NOT NULL REFERENCES {chats}(id) ON DELETE CASCADE,
    seq        BIGINT NOT NULL,
    id         TEXT NOT NULL,
    run_id     TEXT,
    type       TEXT NOT NULL,
    step       INTEGER,
    data       JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (chat_id, seq)
);
CREATE INDEX IF NOT EXISTS {p}events_type ON {events} (chat_id, type);
"""),
]

_CHAT_COLUMNS = (
    "id, agent_id, owner_id, title, status, metadata, message_count, input_tokens, output_tokens, cost_usd, "
    "version, forked_from_id, forked_from_seq, created_at, updated_at, last_message_at, archived_at, deleted_at, "
    "lease_expires_at"
)


class PostgresStore:
    """A `PersistenceStore` on PostgreSQL. Give it a DSN (it creates a pool on first use) or a pool."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: asyncpg.Pool | None = None,
        schema: str = "public",
        table_prefix: str = "harness_",
        min_size: int = 1,
        max_size: int = 10,
        migrate: bool = True,
    ) -> None:
        if dsn is None and pool is None:
            raise ValueError("PostgresStore needs a dsn or a pool")
        for name, value in (("schema", schema), ("table_prefix", table_prefix.rstrip("_") or "x")):
            if not _IDENTIFIER.match(value):
                raise ValueError(f"{name} must be lowercase letters, digits and underscores, got {value!r}")
        self._dsn, self._pool, self._own_pool = dsn, pool, pool is None
        self._min_size, self._max_size, self._migrate = min_size, max_size, migrate
        self._ready = False
        self._lock = asyncio.Lock()
        self.prefix = table_prefix
        q = lambda table: f'"{schema}"."{table_prefix}{table}"'  # noqa: E731
        self.schema = schema
        self.chats, self.messages, self.runs, self.events = q("chats"), q("messages"), q("runs"), q("events")
        self.migrations_table = q("schema_migrations")

    # --- connection and schema ------------------------------------------------------

    async def _get_pool(self) -> asyncpg.Pool:
        if self._ready and self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                try:
                    self._pool = await asyncpg.create_pool(self._dsn, min_size=self._min_size, max_size=self._max_size,
                                                           init=_init_connection)
                except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                    raise PersistenceError(f"could not connect to PostgreSQL: {exc}") from exc
            if not self._ready:
                if self._migrate:
                    await self.migrate()
                self._ready = True
        return self._pool

    async def migrate(self) -> list[int]:
        """Apply missing schema versions. Returns the versions applied now."""
        pool = self._pool or await self._get_pool()
        applied: list[int] = []
        names = {"chats": self.chats, "messages": self.messages, "runs": self.runs, "events": self.events,
                 "p": self.prefix}
        try:
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK)
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                await conn.execute(f"CREATE TABLE IF NOT EXISTS {self.migrations_table} "
                                   "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
                done = {r["version"] for r in await conn.fetch(f"SELECT version FROM {self.migrations_table}")}
                for version, sql in MIGRATIONS:
                    if version not in done:
                        await conn.execute(sql.format(**names))
                        await conn.execute(f"INSERT INTO {self.migrations_table} (version) VALUES ($1)", version)
                        applied.append(version)
        except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise PersistenceError(f"could not create the harness tables: {exc}") from exc
        return applied

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[asyncpg.Connection]:
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                yield conn
        except HarnessError:
            raise
        except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise PersistenceError(f"database error: {exc}") from exc

    async def close(self) -> None:
        if self._pool is not None and self._own_pool:
            await self._pool.close()
            self._pool, self._ready = None, False

    # --- used by the runtime --------------------------------------------------------

    async def acquire(
        self, chat_id: str, *, agent_id: str, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Lease:
        token = new_id("lease_")
        async with self._conn() as conn, conn.transaction():
            created = await conn.fetchval(
                f"INSERT INTO {self.chats} (id, agent_id, owner_id, status, metadata) VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (id) DO NOTHING RETURNING true",
                chat_id, agent_id, owner_id, RunStatus.PENDING.value, metadata or {},
            )
            row = await conn.fetchrow(
                f"SELECT agent_id, owner_id, status, state, message_count, deleted_at, lease_expires_at, now() AS db_now "
                f"FROM {self.chats} WHERE id = $1 FOR UPDATE", chat_id,
            )
            check_claim(chat_id, stored_agent_id=row["agent_id"], stored_owner_id=row["owner_id"],
                        deleted_at=row["deleted_at"], lease_expires_at=row["lease_expires_at"],
                        agent_id=agent_id, owner_id=owner_id, now=row["db_now"])
            await conn.execute(
                f"UPDATE {self.chats} SET lease_token = $2, lease_expires_at = now() + make_interval(secs => $3), "
                "owner_id = COALESCE(owner_id, $4), version = version + 1 WHERE id = $1",
                chat_id, token, float(lease_seconds), owner_id,
            )
            state = None
            last_id = None
            if row["state"]:
                messages = await conn.fetch(f"SELECT id, content FROM {self.messages} WHERE chat_id = $1 ORDER BY seq",
                                            chat_id)
                state = restore_state(row["state"], [m["content"] for m in messages])
                last_id = messages[-1]["id"] if messages else None
        return Lease(
            chat_id=chat_id, token=token, lease_seconds=lease_seconds, created=bool(created),
            recovered=row["status"] == RunStatus.RUNNING.value, state=state,
            message_count=row["message_count"], last_message_id=last_id,
        )

    async def save(
        self, lease: Lease, state: AgentState, run: RunRecord, *, events: Sequence[Event] = (), release: bool = False,
    ) -> int:
        async with self._conn() as conn, conn.transaction():
            row = await conn.fetchrow(f"SELECT lease_token, message_count FROM {self.chats} WHERE id = $1 FOR UPDATE",
                                      lease.chat_id)
            if row is None or row["lease_token"] != lease.token:
                raise PersistenceConflictError(
                    f"lost the lease on chat '{lease.chat_id}' (it expired and another run took the chat over)"
                )
            start = row["message_count"]
            last_id = (await conn.fetchval(f"SELECT id FROM {self.messages} WHERE chat_id = $1 AND seq = $2",
                                           lease.chat_id, start - 1)) if start else None
            check_history(start, last_id, state)
            new = state.messages[start:]
            if new:
                await conn.executemany(
                    f"INSERT INTO {self.messages} (chat_id, seq, id, run_id, role, text, content, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                    [(lease.chat_id, seq, m.id, run.id, m.role.value, m.text, m.model_dump(mode="json"), m.created_at)
                     for seq, m in enumerate(new, start=start)],
                )
            await conn.execute(
                f"INSERT INTO {self.runs} (id, chat_id, kind, status, recovered, first_seq, last_seq, steps, input_tokens, "
                "output_tokens, cost_usd, error, started_at, finished_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
                "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, last_seq = EXCLUDED.last_seq, "
                "steps = EXCLUDED.steps, input_tokens = EXCLUDED.input_tokens, output_tokens = EXCLUDED.output_tokens, "
                "cost_usd = EXCLUDED.cost_usd, error = EXCLUDED.error, finished_at = EXCLUDED.finished_at",
                run.id, lease.chat_id, run.kind, run.status.value, run.recovered, run.first_seq, run.last_seq, run.steps,
                run.input_tokens, run.output_tokens, run.cost_usd,
                run.error.model_dump(mode="json") if run.error else None, run.started_at, run.finished_at,
            )
            if events:
                await conn.executemany(
                    f"INSERT INTO {self.events} (chat_id, seq, id, run_id, type, step, data, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (chat_id, seq) DO NOTHING",
                    [(lease.chat_id, e.seq, e.id, run.id, e.type, e.step, event_row(e)["data"], e.timestamp)
                     for e in events],
                )
            return await conn.fetchval(
                f"UPDATE {self.chats} SET status = $2, state = $3, message_count = $4, title = COALESCE(title, $5), "
                "input_tokens = $6, output_tokens = $7, cost_usd = $8, version = version + 1, updated_at = now(), "
                "last_message_at = CASE WHEN $9 THEN now() ELSE last_message_at END, "
                "lease_token = CASE WHEN $10 THEN NULL ELSE lease_token END, "
                "lease_expires_at = CASE WHEN $10 THEN NULL ELSE now() + make_interval(secs => $11) END "
                "WHERE id = $1 RETURNING version",
                lease.chat_id, state.status.value, state_snapshot(state), len(state.messages),
                derive_title(state.messages), state.usage.input_tokens, state.usage.output_tokens, state.usage.cost_usd,
                bool(new), release, float(lease.lease_seconds),
            )

    async def renew(self, lease: Lease) -> None:
        async with self._conn() as conn:
            status = await conn.execute(
                f"UPDATE {self.chats} SET lease_expires_at = now() + make_interval(secs => $3) "
                "WHERE id = $1 AND lease_token = $2", lease.chat_id, lease.token, float(lease.lease_seconds),
            )
        if status.endswith(" 0"):
            raise PersistenceConflictError(f"lost the lease on chat '{lease.chat_id}'")

    async def release(self, lease: Lease) -> None:
        async with self._conn() as conn:
            if lease.created:
                deleted = await conn.execute(
                    f"DELETE FROM {self.chats} WHERE id = $1 AND lease_token = $2 AND message_count = 0 "
                    "AND state = '{}'::jsonb", lease.chat_id, lease.token,
                )
                if not deleted.endswith(" 0"):
                    return  # created by this lease and never saved
            await conn.execute(
                f"UPDATE {self.chats} SET lease_token = NULL, lease_expires_at = NULL WHERE id = $1 AND lease_token = $2",
                lease.chat_id, lease.token,
            )

    # --- reading ----------------------------------------------------------------------

    async def get_chat(self, chat_id: str) -> ChatRecord | None:
        async with self._conn() as conn:
            row = await conn.fetchrow(f"SELECT {_CHAT_COLUMNS} FROM {self.chats} WHERE id = $1", chat_id)
        return _record(row) if row else None

    async def list_chats(
        self, *, owner_id: str | None = None, status: RunStatus | str | None = None,
        metadata: dict[str, Any] | None = None, include_archived: bool = False, limit: int = 50,
        cursor: str | None = None,
    ) -> ChatPage:
        conditions, args = ["deleted_at IS NULL"], []

        def arg(value: Any) -> str:
            args.append(value)
            return f"${len(args)}"

        if owner_id is not None:
            conditions.append(f"owner_id = {arg(owner_id)}")
        if status is not None:
            conditions.append(f"status = {arg(RunStatus(status).value)}")
        if not include_archived:
            conditions.append("archived_at IS NULL")
        if metadata:
            conditions.append(f"metadata @> {arg(metadata)}::jsonb")
        if cursor:
            stamp, last_id = decode_cursor(cursor)
            conditions.append(f"(updated_at, id) < ({arg(stamp)}, {arg(last_id)})")
        sql = (f"SELECT {_CHAT_COLUMNS} FROM {self.chats} WHERE {' AND '.join(conditions)} "
               f"ORDER BY updated_at DESC, id DESC LIMIT {arg(limit + 1)}")
        async with self._conn() as conn:
            rows = await conn.fetch(sql, *args)
        items = [_record(r) for r in rows[:limit]]
        return ChatPage(items=items, next_cursor=encode_cursor(items[-1]) if len(rows) > limit else None)

    async def load_state(self, chat_id: str) -> AgentState | None:
        async with self._conn() as conn:
            snapshot = await conn.fetchval(f"SELECT state FROM {self.chats} WHERE id = $1", chat_id)
            if not snapshot:
                return None
            rows = await conn.fetch(f"SELECT content FROM {self.messages} WHERE chat_id = $1 ORDER BY seq", chat_id)
        return restore_state(snapshot, [r["content"] for r in rows])

    async def get_messages(
        self, chat_id: str, *, before_seq: int | None = None, limit: int | None = None,
    ) -> list[StoredMessage]:
        async with self._conn() as conn:
            rows = await conn.fetch(
                f"SELECT seq, run_id, content FROM {self.messages} WHERE chat_id = $1 "
                "AND ($2::int IS NULL OR seq < $2) ORDER BY seq DESC LIMIT $3",
                chat_id, before_seq, limit,
            )
        return [StoredMessage(chat_id=chat_id, seq=r["seq"], run_id=r["run_id"],
                              message=Message.model_validate(r["content"])) for r in reversed(rows)]

    async def list_runs(self, chat_id: str) -> list[RunRecord]:
        async with self._conn() as conn:
            rows = await conn.fetch(
                f"SELECT id, chat_id, kind, status, recovered, first_seq, last_seq, steps, input_tokens, output_tokens, "
                f"cost_usd, error, started_at, finished_at FROM {self.runs} WHERE chat_id = $1 ORDER BY started_at",
                chat_id,
            )
        return [RunRecord.model_validate(dict(r)) for r in rows]

    async def get_events(self, chat_id: str, *, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        async with self._conn() as conn:
            rows = await conn.fetch(
                f"SELECT e.seq, e.id, e.type, e.step, e.data, e.created_at, c.agent_id FROM {self.events} e "
                f"JOIN {self.chats} c ON c.id = e.chat_id WHERE e.chat_id = $1 AND e.seq > $2 ORDER BY e.seq LIMIT $3",
                chat_id, after_seq, limit,
            )
        return [Event(id=r["id"], type=r["type"], seq=r["seq"], timestamp=r["created_at"], execution_id=chat_id,
                      agent_id=r["agent_id"], step=r["step"], data=r["data"]) for r in rows]

    # --- managing chats -------------------------------------------------------------

    async def create_chat(
        self, state: AgentState, *, owner_id: str | None = None, metadata: dict[str, Any] | None = None,
        title: str | None = None, forked_from: tuple[str, int] | None = None,
    ) -> ChatRecord:
        async with self._conn() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"INSERT INTO {self.chats} (id, agent_id, owner_id, title, status, metadata, state, message_count, "
                "input_tokens, output_tokens, cost_usd, version, forked_from_id, forked_from_seq, last_message_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 1, $12, $13, CASE WHEN $8 > 0 THEN now() END) "
                f"ON CONFLICT (id) DO NOTHING RETURNING {_CHAT_COLUMNS}",
                state.execution_id, state.agent_id, owner_id, title or derive_title(state.messages), state.status.value,
                metadata or {}, state_snapshot(state), len(state.messages), state.usage.input_tokens,
                state.usage.output_tokens, state.usage.cost_usd,
                forked_from[0] if forked_from else None, forked_from[1] if forked_from else None,
            )
            if row is None:
                raise PersistenceConflictError(f"chat '{state.execution_id}' already exists")
            if state.messages:
                await conn.executemany(
                    f"INSERT INTO {self.messages} (chat_id, seq, id, run_id, role, text, content, created_at) "
                    "VALUES ($1, $2, $3, NULL, $4, $5, $6, $7)",
                    [(state.execution_id, seq, m.id, m.role.value, m.text, m.model_dump(mode="json"), m.created_at)
                     for seq, m in enumerate(state.messages)],
                )
        return _record(row)

    async def update_chat(
        self, chat_id: str, *, title: str | None = None, metadata: dict[str, Any] | None = None,
        archived: bool | None = None,
    ) -> ChatRecord:
        async with self._conn() as conn:
            row = await conn.fetchrow(
                f"UPDATE {self.chats} SET title = COALESCE($2, title), metadata = metadata || $3::jsonb, "
                "archived_at = CASE WHEN $4::boolean IS NULL THEN archived_at "
                "WHEN $4 THEN COALESCE(archived_at, now()) ELSE NULL END, version = version + 1 "
                f"WHERE id = $1 AND deleted_at IS NULL RETURNING {_CHAT_COLUMNS}",
                chat_id, title, metadata or {}, archived,
            )
        if row is None:
            raise ValidationError(f"no chat '{chat_id}'", code=ErrorCode.INVALID_CONFIG)
        return _record(row)

    async def delete_chat(self, chat_id: str, *, hard: bool = False) -> bool:
        async with self._conn() as conn:
            if hard:
                status = await conn.execute(f"DELETE FROM {self.chats} WHERE id = $1", chat_id)
            else:
                status = await conn.execute(
                    f"UPDATE {self.chats} SET deleted_at = COALESCE(deleted_at, now()), version = version + 1 "
                    "WHERE id = $1", chat_id,
                )
        return not status.endswith(" 0")


async def _init_connection(conn: asyncpg.Connection) -> None:
    # JSON columns in and out as Python objects.
    for name in ("json", "jsonb"):
        await conn.set_type_codec(name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


def _record(row: asyncpg.Record) -> ChatRecord:
    data: dict[str, Any] = dict(row)
    return ChatRecord.model_validate(data)


__all__ = ["MIGRATIONS", "PostgresStore"]
