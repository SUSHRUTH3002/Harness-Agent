"""Postgres access for the web UI: plain SQL over an asyncpg pool.

The harness itself stays storage-free (state lives in memory during a run).
This module is where a session's `AgentState` is saved after each run and
rebuilt before the next one:

    sessions.state  <- AgentState without messages/tool_results (step, usage, extensions, ...)
    messages        <- every Message, in order, exactly as the harness produced it
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from agent_harness import AgentState, Message, RunResult

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
OWNER_TYPES = ("user", "incident")
# What sessions.state leaves out; those live in their own tables (or aren't needed between runs).
_STATE_EXCLUDE = {"messages", "tool_results"}


async def _init_connection(conn: asyncpg.Connection) -> None:
    # jsonb in and out as plain Python values.
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


def _row(record: asyncpg.Record | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: (str(v) if isinstance(v, UUID) else v) for k, v in record.items()}


def _rows(records: list[asyncpg.Record]) -> list[dict[str, Any]]:
    return [_row(r) for r in records]


class Database:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, url: str) -> Database:
        pool = await asyncpg.create_pool(url, min_size=1, max_size=10, init=_init_connection)
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_FILE.read_text())
        logger.info("database ready (schema applied)")
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    # --- owners --------------------------------------------------------------------------

    async def list_users(self) -> list[dict[str, Any]]:
        return _rows(await self.pool.fetch("SELECT * FROM users ORDER BY created_at"))

    async def create_user(self, name: str, email: str | None) -> dict[str, Any]:
        return _row(await self.pool.fetchrow(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *", name, email))

    async def list_incidents(self) -> list[dict[str, Any]]:
        return _rows(await self.pool.fetch("SELECT * FROM incidents ORDER BY created_at DESC"))

    async def create_incident(self, title: str, severity: str, description: str | None) -> dict[str, Any]:
        return _row(await self.pool.fetchrow(
            "INSERT INTO incidents (title, severity, description) VALUES ($1, $2, $3) RETURNING *",
            title, severity, description))

    # --- sessions -------------------------------------------------------------------------

    _SESSION_COLUMNS = "id, owner_type, owner_id, title, created_at, updated_at"

    async def list_sessions(self, owner_type: str, owner_id: str) -> list[dict[str, Any]]:
        return _rows(await self.pool.fetch(
            f"SELECT {self._SESSION_COLUMNS} FROM sessions WHERE owner_type = $1 AND owner_id = $2 "
            "ORDER BY updated_at DESC", owner_type, UUID(owner_id)))

    async def create_session(self, owner_type: str, owner_id: str, title: str | None) -> dict[str, Any]:
        return _row(await self.pool.fetchrow(
            f"INSERT INTO sessions (owner_type, owner_id, title) VALUES ($1, $2, COALESCE($3, 'New chat')) "
            f"RETURNING {self._SESSION_COLUMNS}", owner_type, UUID(owner_id), title))

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return _row(await self.pool.fetchrow(
            f"SELECT {self._SESSION_COLUMNS}, state FROM sessions WHERE id = $1", UUID(session_id)))

    async def rename_session(self, session_id: str, title: str) -> dict[str, Any] | None:
        return _row(await self.pool.fetchrow(
            f"UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 RETURNING {self._SESSION_COLUMNS}",
            UUID(session_id), title))

    async def delete_session(self, session_id: str) -> bool:
        return (await self.pool.execute("DELETE FROM sessions WHERE id = $1", UUID(session_id))) != "DELETE 0"

    async def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        return _rows(await self.pool.fetch(
            "SELECT seq, id, run_id, role, text, body, created_at FROM messages WHERE session_id = $1 ORDER BY seq",
            UUID(session_id)))

    async def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        runs = _rows(await self.pool.fetch(
            "SELECT * FROM runs WHERE session_id = $1 ORDER BY started_at", UUID(session_id)))
        events = _rows(await self.pool.fetch(
            "SELECT e.run_id, e.seq, e.type, e.step, e.data, e.at FROM run_events e JOIN runs r ON r.id = e.run_id "
            "WHERE r.session_id = $1 ORDER BY e.run_id, e.seq", UUID(session_id)))
        by_run: dict[str, list[dict[str, Any]]] = {}
        for e in events:
            by_run.setdefault(e.pop("run_id"), []).append(e)
        for run in runs:
            run["events"] = by_run.get(run["id"], [])
        return runs

    # --- agent state ----------------------------------------------------------------------

    async def load_state(self, session_id: str) -> AgentState | None:
        """Rebuild the session's AgentState, or None if it has never run."""
        async with self.pool.acquire() as conn:
            snapshot = await conn.fetchval("SELECT state FROM sessions WHERE id = $1", UUID(session_id))
            if snapshot is None:
                return None
            bodies = await conn.fetch("SELECT body FROM messages WHERE session_id = $1 ORDER BY seq",
                                      UUID(session_id))
        return AgentState.model_validate({**snapshot, "messages": [r["body"] for r in bodies]})

    # --- runs -----------------------------------------------------------------------------

    async def create_run(self, run_id: str, session_id: str, input: str) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("INSERT INTO runs (id, session_id, input) VALUES ($1, $2, $3)",
                               UUID(run_id), UUID(session_id), input)
            # First message names the chat.
            await conn.execute(
                "UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 AND title = 'New chat'",
                UUID(session_id), _title_from(input))

    async def save_run(
        self, run_id: str, session_id: str, result: RunResult, first_new: int, events: list[dict[str, Any]]
    ) -> None:
        """Persist everything a finished run produced, in one transaction."""
        state = result.state
        new_messages: list[Message] = state.messages[first_new:]
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "INSERT INTO messages (session_id, seq, id, run_id, role, text, body) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                [
                    (UUID(session_id), first_new + i, m.id, UUID(run_id), m.role.value, m.text,
                     m.model_dump(mode="json"))
                    for i, m in enumerate(new_messages)
                ],
            )
            await conn.execute(
                "UPDATE sessions SET state = $2, updated_at = now() WHERE id = $1",
                UUID(session_id), state.model_dump(mode="json", exclude=_STATE_EXCLUDE),
            )
            await conn.execute(
                "UPDATE runs SET status = $2, output = $3, error = $4, usage = $5, steps = $6, finished_at = now() "
                "WHERE id = $1",
                UUID(run_id), result.status.value, result.output,
                result.error.model_dump(mode="json") if result.error else None,
                result.usage.model_dump(), result.steps,
            )
            await conn.executemany(
                "INSERT INTO run_events (run_id, seq, type, step, data, at) VALUES ($1, $2, $3, $4, $5, $6)",
                [(UUID(run_id), e["seq"], e["type"], e["step"], e["data"], datetime.fromisoformat(e["at"])) for e in events],
            )

    async def fail_run(self, run_id: str, message: str) -> None:
        await self.pool.execute(
            "UPDATE runs SET status = 'failed', error = $2, finished_at = now() WHERE id = $1 AND finished_at IS NULL",
            UUID(run_id), {"code": "PERSISTENCE_ERROR", "message": message})

    async def fail_unfinished_runs(self) -> int:
        """Runs still 'running' from a previous server process can never finish; mark them."""
        status = await self.pool.execute(
            "UPDATE runs SET status = 'failed', finished_at = now(), "
            "error = '{\"code\": \"SERVER_RESTARTED\", \"message\": \"The server stopped during this run.\"}' "
            "WHERE finished_at IS NULL")
        return int(status.split()[-1])


def _title_from(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
