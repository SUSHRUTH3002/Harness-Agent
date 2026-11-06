"""Web UI backend: FastAPI over the harness, with users/incidents, sessions and chats in Postgres.

    .venv/bin/uvicorn server.app:app --reload --port 8000

Uses the same .env and the same agent/runtime as main.py (main.load_config + main.build),
plus HARNESS_DATABASE_URL. If ui/dist exists (npm run build), it is served at /.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_harness import FileArtifactStore, ToolError
from main import ROOT, ConfigError, _env, build, load_config, load_env_file, setup_logging
from server.db import Database
from server.runs import RunConflict, RunManager

logger = logging.getLogger("server")

DEFAULT_DATABASE_URL = "postgresql://harness:harness@localhost:5433/harness"
UI_DIST = ROOT / "ui" / "dist"
# SSE comment sent while a run is quiet, so proxies don't close the connection.
_KEEPALIVE_SECONDS = 15


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_env_file(ROOT / ".env")
    setup_logging(_env("HARNESS_LOG_LEVEL") or "INFO", _env("HARNESS_LOG_FILE"))
    logging.getLogger("server").setLevel(logging.getLogger("agent_harness").level)
    try:
        config = load_config()
    except ConfigError as exc:
        raise RuntimeError(f"config error: {exc}") from exc
    runtime, agent = build(config)
    url = _env("HARNESS_DATABASE_URL") or DEFAULT_DATABASE_URL
    try:
        db = await Database.connect(url)
    except Exception as exc:
        raise RuntimeError(
            f"could not connect to Postgres at {url.split('@')[-1]} ({exc}). Is it running? `docker compose up -d`"
        ) from exc
    stale = await db.fail_unfinished_runs()
    if stale:
        logger.warning("marked %d run(s) left unfinished by a previous server process as failed", stale)

    app.state.db = db
    app.state.config = config
    app.state.agent = agent
    app.state.runs = RunManager(db, runtime, agent)
    app.state.artifacts = FileArtifactStore(config["artifact_dir"])
    logger.info("server ready: model=%s database=%s", config["model"], url.split("@")[-1])
    try:
        yield
    finally:
        await app.state.runs.shutdown()
        await db.close()


app = FastAPI(title="agent-harness", lifespan=lifespan)


def _db() -> Database:
    return app.state.db


def _check_uuid(value: str, what: str) -> str:
    try:
        UUID(value)
    except ValueError:
        raise HTTPException(404, f"{what} not found") from None
    return value


# --- request bodies --------------------------------------------------------------------------

class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str | None = None


class IncidentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    severity: str = "sev3"
    description: str | None = None


class SessionIn(BaseModel):
    owner_type: Literal["user", "incident"]
    owner_id: str
    title: str | None = None


class SessionPatch(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class MessageIn(BaseModel):
    content: str = Field(min_length=1)


# --- routes ------------------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, str]:
    await _db().pool.fetchval("SELECT 1")
    return {"status": "ok"}


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    agent = app.state.agent
    limits = agent.limits
    config = app.state.config
    return {
        "model": config["model"],
        "provider": "local" if config["local"] else "litellm",
        "tools": [t.name for t in agent.tools],
        "max_steps": limits.max_steps,
        "max_context_tokens": limits.max_context_tokens,
        "large_result_handling": config["large_result_handling"],
        "compaction": config["compaction"],
        "compaction_trigger_ratio": limits.compaction_trigger_ratio,
    }


@app.get("/api/users")
async def list_users() -> list[dict[str, Any]]:
    return await _db().list_users()


@app.post("/api/users", status_code=201)
async def create_user(body: UserIn) -> dict[str, Any]:
    try:
        return await _db().create_user(body.name.strip(), (body.email or "").strip() or None)
    except Exception as exc:
        if "users_email_key" in str(exc):
            raise HTTPException(409, "A user with that email already exists.") from None
        raise


@app.get("/api/incidents")
async def list_incidents() -> list[dict[str, Any]]:
    return await _db().list_incidents()


@app.post("/api/incidents", status_code=201)
async def create_incident(body: IncidentIn) -> dict[str, Any]:
    return await _db().create_incident(body.title.strip(), body.severity, body.description)


@app.get("/api/sessions")
async def list_sessions(owner_type: Literal["user", "incident"], owner_id: str = Query()) -> list[dict[str, Any]]:
    return await _db().list_sessions(owner_type, _check_uuid(owner_id, "owner"))


@app.post("/api/sessions", status_code=201)
async def create_session(body: SessionIn) -> dict[str, Any]:
    _check_uuid(body.owner_id, body.owner_type)
    try:
        return await _db().create_session(body.owner_type, body.owner_id, body.title)
    except Exception as exc:
        if "violates foreign key constraint" in str(exc):
            raise HTTPException(404, f"{body.owner_type} not found") from None
        raise


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = await _db().get_session(_check_uuid(session_id, "session"))
    if session is None:
        raise HTTPException(404, "session not found")
    state = session.pop("state") or {}
    compaction = (state.get("extensions") or {}).get("compaction") or {}
    active = app.state.runs.active_run(session_id)
    return {
        **session,
        "runs": await _db().list_runs(session_id),
        "active_run_id": active.id if active else None,
        # What the agent is carrying between turns -- shown in the UI's memory panel.
        "memory": {
            "execution_id": state.get("execution_id"),
            "steps": state.get("step", 0),
            "usage": state.get("usage"),
            "compactions": compaction.get("count", 0),
            "covered_until": compaction.get("covered_until"),
            "summary": compaction.get("summary"),
            "archives": compaction.get("archives", []),
            "compaction_history": compaction.get("history", []),
        },
    }


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str) -> list[dict[str, Any]]:
    """The raw transcript: every message the harness keeps for this session."""
    return await _db().list_messages(_check_uuid(session_id, "session"))


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, body: SessionPatch) -> dict[str, Any]:
    session = await _db().rename_session(_check_uuid(session_id, "session"), body.title.strip())
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    if app.state.runs.active_run(session_id):
        raise HTTPException(409, "Stop the running answer before deleting this chat.")
    if not await _db().delete_session(_check_uuid(session_id, "session")):
        raise HTTPException(404, "session not found")


@app.post("/api/sessions/{session_id}/messages", status_code=202)
async def send_message(session_id: str, body: MessageIn) -> dict[str, str]:
    if await _db().get_session(_check_uuid(session_id, "session")) is None:
        raise HTTPException(404, "session not found")
    try:
        live = await app.state.runs.start(session_id, body.content.strip())
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from None
    return {"run_id": live.id}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, bool]:
    return {"cancelled": app.state.runs.cancel(run_id)}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    """Server-sent events: every event of the run so far, then live ones until it's saved."""
    live = app.state.runs.runs.get(run_id)
    if live is None:
        raise HTTPException(404, "run is not live; its events are in GET /api/sessions/{id}")

    async def stream() -> AsyncIterator[str]:
        queue = live.subscribe()
        try:
            while True:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if record is None:
                    yield "event: end\ndata: {}\n\n"
                    return
                yield f"data: {json.dumps(record, default=str)}\n\n"
        finally:
            live.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/artifacts/{artifact_id}", response_class=PlainTextResponse)
async def read_artifact(artifact_id: str) -> str:
    """An offloaded tool result or a compaction archive, as plain text."""
    try:
        return await app.state.artifacts.read(artifact_id)
    except ToolError:
        raise HTTPException(404, "artifact not found") from None


# Built UI (npm run build), if present. In development, Vite serves the UI and proxies /api here.
if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
