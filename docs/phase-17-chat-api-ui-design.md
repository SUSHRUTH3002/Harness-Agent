# Phase 17: Chat API and Web UI

Status: **17a (Chat API) implemented. 17b (Web UI) designed, not started.**

> Numbering: while this was being designed it was called "Phase 11a/11b" in conversation. Phase 11 in the
> roadmap is Subagents, so the Chat API and UI are recorded as Phase 17.

## Goal

A web chat on top of the harness: send a message, watch the answer being written, see tool calls, answer
the agent's questions and approvals (HITL), and reopen any saved chat. The API is plain HTTP + WebSocket, so
any client can use it, not only the web page.

```
Browser / any client
   │  REST: chats, messages, answers, cancel        WebSocket: one run's events, live
   ▼
create_app()  (FastAPI, api/server.py)  ── thin routes: auth, owner, validation, errors
   ▼
ChatService   (api/service.py)          ── framework-free logic; scripts can drive it directly
   │  runtime.start(..., observers=[RunBroadcast])      store.get_chat / list / messages
   ▼
Runtime ── PersistenceStore (Postgres)        RunBroadcast (api/broadcast.py): the active run's events,
                                              replayable, any number of readers
```

## 17a: what is built

| File | Contents |
|---|---|
| `api/models.py` | Request and response shapes (pydantic only). |
| `api/broadcast.py` | `RunBroadcast`: one run's events kept for any number of readers, replayed from a `seq`. |
| `api/service.py` | `ChatService`: chats, runs, ownership, the active-run table, shutdown. |
| `api/server.py` | `create_app(runtime, agent, ...)`: routes, bearer token, owner header, CORS, WebSocket. |
| `api_main.py` | Entry point (`python api_main.py`); reuses `main.py`'s config and `build()`. |
| `pyproject.toml`, `.env.example` | `api` extra (FastAPI, uvicorn) and the `HARNESS_API_*` settings. |

The core is untouched: the API only calls `Runtime.start()` (with `observers=`, `answers=`), `Execution.cancel()`
and the `PersistenceStore`. `agent_harness.api` is not imported by `agent_harness/__init__.py`, so the core still
depends only on pydantic.

### Routes

| Route | Does |
|---|---|
| `GET /api/health` | Up? Active run count. No auth. |
| `POST /api/chats` | Create an empty chat (`title`, `metadata`) → 201. |
| `GET /api/chats` | List, most recently active first. `status`, `include_archived`, `limit`, `cursor`. |
| `GET /api/chats/{id}` | Latest messages (`limit`, `before_seq` to page back), `has_more`, `pending_interactions`, `streaming`. |
| `PATCH /api/chats/{id}` | Rename, archive, merge metadata. |
| `DELETE /api/chats/{id}` | Soft delete. Refused while running. |
| `POST /api/chats/{id}/messages` | `{"text": ...}` → 202 once the run has started (chat leased, message saved). |
| `POST /api/chats/{id}/answers` | `{"answers": [{interaction_id, action, arguments?, value?, note?}]}` → 202; resumes a suspended run. |
| `POST /api/chats/{id}/cancel` | Stop the active run (saved as `cancelled`). |
| `WS /api/chats/{id}/stream?after_seq=N` | The run's events, live. |

Errors are `{"error": {"code", "message"}}`: 404 not found (or someone else's), 409 busy / not waiting for
answers, 422 invalid, 429 server at its run limit, 401 bad token, 503 database trouble.

### The stream

Each frame is one event exactly as in `events.py` (`{"type": "llm.token", "seq": 12, "data": {...}, "transient":
true, "timestamp": "...ISO..."}`), or a control frame:

| Frame | Meaning |
|---|---|
| `stream.idle` | Nothing is running on this server; carries the chat's saved `status`. Read the chat instead. |
| `stream.gap` | Streamed pieces were dropped (the client read too slowly or reconnected too late). Reload the chat. |
| `stream.end` | The run ended: `status` and `error`. The server closes. |

A client that connects while the run is going gets everything since `after_seq` (default 0: from the start of
the run), then the live part. After a dropped connection, reconnect with the last `seq` seen. Closing the socket
never stops the run.

### Decisions (and how they differ from the first sketch)

| # | Decision | Why |
|---|---|---|
| 1 | The run is addressed by the **chat id**; there is no separate `execution_id` in URLs. | In the harness a chat *is* its `execution_id`, and only one run is active per chat. |
| 2 | Answers go to **`POST /chats/{id}/answers`** (a list), not `/interactions/{id}/answer`. | Interactions belong to a chat, and several can be open at once; the run resumes when they are answered. |
| 3 | Added `cancel`, `PATCH`, `DELETE`, `health`; no fork route yet. | A chat UI needs Stop, rename, archive and delete; the store already supports them. Fork (edit/regenerate) is a later addition. |
| 4 | Live events come from a per-run **`RunBroadcast`** attached as an observer, not from `Execution.events()`. | `events()` has one reader and forgets what it delivered; the UI needs several tabs, reconnects and late joiners. |
| 5 | `POST .../messages` waits for `run.started` before answering. | "Chat busy" (another process), bad answers and a lost database then come back as a real 409/422/503, not as a silent failed stream. |
| 6 | Split into `service.py` (no framework) and `server.py`, instead of `handlers.py` / `websocket.py`. | The logic is testable without HTTP, and a second front end (gRPC, Slack) could reuse it. |
| 7 | Not exported from `agent_harness/__init__.py`. | It would import FastAPI into the core. |
| 8 | Default listen address is `127.0.0.1`; `HARNESS_API_TOKEN` enables bearer auth. | Safe by default; open to the network only on purpose. |
| 9 | No `HARNESS_API_ENABLED` / `HARNESS_UI_*` settings. | The API is its own entry point; UI settings belong to 17b. |

### Who is calling

- **Token**: with `HARNESS_API_TOKEN` set, every route but `/health` needs `Authorization: Bearer <token>`.
  A WebSocket from a browser (which cannot set headers) sends `{"type": "auth", "token": "..."}` as its first frame.
- **Owner**: `X-Owner-Id` (or `?owner_id=` on the WebSocket) says whose chats the caller may see. The API
  **trusts** it, so deploy behind whatever authenticates your users and sets the header. A chat with an owner is
  "not found" to other owners; no header means no restriction. User login and per-user tokens are not built here.
- **Origin**: WebSocket handshakes from a browser are accepted only from the server's own origin or
  `HARNESS_API_CORS_ORIGINS` (CORS does not cover WebSockets).

### Limits and behaviour worth knowing

- **One run per chat**: a second message while one is running is a 409. Across processes the store's lease enforces it.
- **Active runs live in one process's memory.** With several server processes, route a chat to the same one (sticky
  routing); a client that lands elsewhere gets `stream.idle` and reads the saved chat. Saved data is always complete.
- **`HARNESS_API_MAX_ACTIVE_RUNS`** (default 50) bounds concurrent model calls; beyond it, 429.
- **Buffer**: each active run keeps up to 5000 streamed pieces (durable events are always kept); finished runs stay
  watchable for `HARNESS_API_KEEP_FINISHED` seconds (default 60).
- **Shutdown** cancels active runs; each saves itself as `cancelled`.
- **Messages are limited** to 100,000 characters.
- Without `HARNESS_DATABASE_URL`, `api_main.py` uses an in-memory store (a warning is logged; chats vanish on restart).

### Try it

```bash
pip install -e ".[api,postgres]"
python api_main.py                                # http://127.0.0.1:8000, interactive docs at /docs
curl -s -X POST localhost:8000/api/chats -H 'Content-Type: application/json' -d '{"title":"demo"}'
curl -s -X POST localhost:8000/api/chats/<id>/messages -H 'Content-Type: application/json' -d '{"text":"What is 17*23?"}'
python - <<'EOF'   # watch it
import asyncio, json, websockets
async def go():
    async with websockets.connect("ws://127.0.0.1:8000/api/chats/<id>/stream") as ws:
        async for raw in ws:
            f = json.loads(raw)
            if f["type"] == "llm.token": print(f["data"]["text"], end="", flush=True)
            if f["type"] == "stream.end": print("\n", f["status"])
asyncio.run(go())
EOF
```

## 17b: Web UI (designed, not started)

A single-page chat in `ui/`, served separately (dev server) or from the API as static files. It uses only the API
above:

- Chat list (`GET /chats`), open a chat (`GET /chats/{id}`), "load earlier" with `before_seq`.
- Send: `POST .../messages`, then open `WS .../stream`; append `llm.token` text to a live bubble; show
  `tool.started` / `tool.completed` as a tool row; `llm.stream_reset` clears the live bubble; `stream.end` or `stream.gap` reloads the chat.
- HITL: a chat whose status is `awaiting_input` shows its `pending_interactions` as a form (approve / edit / reject,
  options, free text, note) and posts to `.../answers`.
- Stop button → `cancel`. Rename/archive/delete from the list.
- Framework: the first sketch proposed React; to be confirmed at the 17b review (a no-build page is also enough for this API).

Open for 17b: React vs. a no-build page; serving the UI from the API; login (the API already takes an owner header and a token).
