"""The Chat API as a FastAPI app (Phase 11a). Needs `pip install -e ".[api]"`.

Routes are thin: they check the caller, call `ChatService`, and shape the response.

    GET    /api/health                       is the server up (no auth)
    POST   /api/chats                        create an empty chat
    GET    /api/chats                        list chats, most recently active first
    GET    /api/chats/{id}                   one chat: latest messages, what it is waiting for
    PATCH  /api/chats/{id}                   rename / archive / merge metadata
    DELETE /api/chats/{id}                   delete (hidden, kept in the database)
    POST   /api/chats/{id}/messages          send a message; the agent starts (202)
    POST   /api/chats/{id}/answers           answer what a suspended run asks (202)
    POST   /api/chats/{id}/cancel            stop the active run
    WS     /api/chats/{id}/stream            the run's events, live (replay with ?after_seq=N)

Callers. With `token` set, every route but /health needs `Authorization: Bearer <token>`. Who the
caller *is* comes from the `X-Owner-Id` header (`?owner_id=` on the WebSocket, where browsers cannot
set headers): this API trusts it, so put it behind whatever authenticates your users and sets that
header. No header means no restriction. A chat with an owner is "not found" to other owners.

WebSocket frames are JSON. Each event is sent as it is (`{"type": "llm.token", "seq": 12, "data":
{...}, ...}`, see events.py). Control frames use `stream.*` types:

    stream.idle   nothing is running here; carries the chat's saved `status`
    stream.gap    pieces were dropped because the client read too slowly: reload the chat
    stream.end    the run ended; carries its `status` (and `error`). The server then closes.

Closing the socket never stops the run. Reconnect with `?after_seq=<last seq seen>` to continue.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Sequence
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Path, Query, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from agent_harness.agent import Agent
from agent_harness.api.broadcast import Gap
from agent_harness.api.models import (
    AnswersIn,
    CancelIn,
    ChatCreate,
    ChatDetail,
    ChatList,
    ChatOut,
    ChatUpdate,
    Health,
    MessageCreate,
    RunStarted,
)
from agent_harness.api.service import (
    BusyError,
    ChatService,
    NotFoundError,
    TooManyRunsError,
)
from agent_harness.errors import ErrorCode, HarnessError, PersistenceError, ValidationError
from agent_harness.events import Event
from agent_harness.runtime import Runtime
from agent_harness.state import RunStatus

logger = logging.getLogger(__name__)

ChatId = Annotated[str, Path(min_length=1, max_length=128)]
WS_AUTH_TIMEOUT = 5.0
CLOSE_NOT_FOUND = 4404
CLOSE_UNAUTHORIZED = 4401


def event_json(event: Event) -> str:
    """An event as JSON text with ISO timestamps (Event.to_json writes them in Python's format)."""
    def default(value: Any) -> Any:
        return value.isoformat() if isinstance(value, datetime) else str(value)

    return json.dumps(event.model_dump(mode="python"), default=default, ensure_ascii=False)


def _status_for(exc: HarnessError) -> int:
    if isinstance(exc, NotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, BusyError):
        return status.HTTP_409_CONFLICT
    if isinstance(exc, TooManyRunsError):
        return status.HTTP_429_TOO_MANY_REQUESTS
    if isinstance(exc, ValidationError):
        return 422
    if isinstance(exc, PersistenceError):
        return (status.HTTP_409_CONFLICT if exc.code == ErrorCode.PERSISTENCE_CONFLICT
                else status.HTTP_503_SERVICE_UNAVAILABLE)
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def owner_from(value: str | None) -> str | None:
    value = (value or "").strip()
    return value[:200] or None


async def caller(x_owner_id: Annotated[str | None, Header()] = None) -> str | None:
    return owner_from(x_owner_id)


# Defined at module level: FastAPI resolves the string annotations of route functions from module globals.
Owner = Annotated[str | None, Depends(caller)]


def create_app(
    runtime: Runtime, agent: Agent, *, token: str | None = None, cors_origins: Sequence[str] = (),
    max_active_runs: int | None = 50, keep_finished: float = 60.0, docs: bool = True,
) -> FastAPI:
    """Build the app. `token`: require this bearer token. `cors_origins`: browser origins allowed to call
    the API (also the origins allowed to open the WebSocket, besides the server's own)."""
    service = ChatService(runtime, agent, max_active_runs=max_active_runs, keep_finished=keep_finished)
    allowed_origins = {o.rstrip("/") for o in cors_origins}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await service.shutdown()

    app = FastAPI(title="Agent Harness Chat API", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url=None, openapi_url="/openapi.json" if docs else None)
    app.state.service = service
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=sorted(allowed_origins), allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-Owner-Id"],
        )

    @app.exception_handler(HarnessError)
    async def harness_error(_: Request, exc: HarnessError) -> JSONResponse:
        code = _status_for(exc)
        if code >= 500:
            logger.error("api error [%s]: %s", exc.code, exc.message)
        return JSONResponse(status_code=code, content={"error": {"code": exc.code, "message": exc.message}})

    def token_ok(value: str | None) -> bool:
        return token is None or (value is not None and hmac.compare_digest(value.encode(), token.encode()))

    def bearer(header: str | None) -> str | None:
        return header[7:].strip() if header and header[:7].lower() == "bearer " else None

    async def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not token_ok(bearer(authorization)):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

    @app.get("/api/health", response_model=Health, tags=["meta"])
    async def health() -> Health:
        return Health(active_runs=service.active_runs)

    @api.post("/chats", status_code=201, response_model=ChatOut)
    async def create_chat(body: ChatCreate, owner: Owner) -> ChatOut:
        return await service.create_chat(owner, title=body.title, metadata=body.metadata)

    @api.get("/chats", response_model=ChatList)
    async def list_chats(
        owner: Owner, status: RunStatus | None = None, include_archived: bool = False,
        limit: Annotated[int, Query(ge=1, le=200)] = 50, cursor: str | None = None,
    ) -> ChatList:
        return await service.list_chats(owner, status=status.value if status else None,
                                        include_archived=include_archived, limit=limit, cursor=cursor)

    @api.get("/chats/{chat_id}", response_model=ChatDetail)
    async def get_chat(
        chat_id: ChatId, owner: Owner, limit: Annotated[int, Query(ge=1, le=500)] = 100,
        before_seq: Annotated[int | None, Query(ge=0)] = None,
    ) -> ChatDetail:
        return await service.get_chat(chat_id, owner, limit=limit, before_seq=before_seq)

    @api.patch("/chats/{chat_id}", response_model=ChatOut)
    async def update_chat(chat_id: ChatId, body: ChatUpdate, owner: Owner) -> ChatOut:
        return await service.update_chat(chat_id, owner, body)

    @api.delete("/chats/{chat_id}", status_code=204)
    async def delete_chat(chat_id: ChatId, owner: Owner) -> Response:
        await service.delete_chat(chat_id, owner)
        return Response(status_code=204)

    @api.post("/chats/{chat_id}/messages", status_code=202, response_model=RunStarted)
    async def send_message(chat_id: ChatId, body: MessageCreate, owner: Owner) -> RunStarted:
        await service.send_message(chat_id, owner, body.text)
        return RunStarted(chat_id=chat_id, stream=f"/api/chats/{chat_id}/stream")

    @api.post("/chats/{chat_id}/answers", status_code=202, response_model=RunStarted)
    async def answer(chat_id: ChatId, body: AnswersIn, owner: Owner) -> RunStarted:
        await service.answer(chat_id, owner, body.answers)
        return RunStarted(chat_id=chat_id, stream=f"/api/chats/{chat_id}/stream")

    @api.post("/chats/{chat_id}/cancel", status_code=202)
    async def cancel(chat_id: ChatId, owner: Owner, body: CancelIn | None = None) -> dict[str, bool]:
        if not await service.cancel(chat_id, owner, body.reason if body else None):
            raise BusyError(f"chat '{chat_id}' has no run active on this server")
        return {"cancelled": True}

    app.include_router(api)

    # --- WebSocket -------------------------------------------------------------------------

    def origin_ok(ws: WebSocket) -> bool:
        # Browsers send Origin on WebSocket and CORS does not apply to it, so check it here.
        origin = ws.headers.get("origin")
        if origin is None:
            return True  # not a browser
        origin = origin.rstrip("/")
        return origin in allowed_origins or urlparse(origin).netloc == ws.headers.get("host")

    async def ws_authenticated(ws: WebSocket) -> bool:
        if token_ok(bearer(ws.headers.get("authorization"))):
            return True
        # Browsers cannot set headers on a WebSocket: they authenticate with a first frame instead.
        try:
            first = await asyncio.wait_for(ws.receive_json(), WS_AUTH_TIMEOUT)
        except Exception:
            return False
        return isinstance(first, dict) and first.get("type") == "auth" and token_ok(first.get("token"))

    @app.websocket("/api/chats/{chat_id}/stream")
    async def stream(ws: WebSocket, chat_id: ChatId, after_seq: int = 0, owner_id: str | None = None) -> None:
        if not origin_ok(ws):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await ws.accept()
        if not await ws_authenticated(ws):
            await ws.close(code=CLOSE_UNAUTHORIZED, reason="unauthorized")
            return
        owner = owner_from(ws.headers.get("x-owner-id") or owner_id)
        try:
            broadcast = await service.watch(chat_id, owner)
            if broadcast is None:
                saved = await service.status(chat_id, owner)
        except NotFoundError:
            await ws.close(code=CLOSE_NOT_FOUND, reason="no such chat")
            return
        if broadcast is None:
            await ws.send_json({"type": "stream.idle", "status": saved.value})
            await ws.close()
            return

        async def pump() -> None:
            async for item in broadcast.subscribe(max(after_seq, 0)):
                if isinstance(item, Gap):
                    await ws.send_json({"type": "stream.gap"})
                else:
                    await ws.send_text(event_json(item))
            await ws.send_json({
                "type": "stream.end", "status": broadcast.status.value if broadcast.status else None,
                "error": broadcast.error.model_dump(include={"code", "message"}) if broadcast.error else None,
            })

        async def until_disconnect() -> None:
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass  # clients have nothing to say after authenticating

        tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(until_disconnect())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await ws.close()

    return app
