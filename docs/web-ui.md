# Web UI: users, incidents, sessions and chats

How to run it: see the README's "Web UI" section. This page explains how it works.

## The pieces

```
 Browser (React, ui/)                 FastAPI (server/)                        Harness (src/agent_harness)
 ───────────────────                  ─────────────────                        ───────────────────────────
 Sidebar: owner → chats   ──REST──►   app.py      routes                       Runtime.start(agent, text,
 ChatView: timeline       ◄──SSE───   runs.py     RunManager / LiveRun ──────►   state=…, on_event=…)
 MemoryPanel, viewers     ──REST──►   db.py       asyncpg + SQL                  │ emits Events (events.py)
                                         │                                       ▼
                                         ▼                                    AgentLoop, tools, offload,
                                      Postgres (docker-compose.yml)            compaction
```

- **The harness is unchanged in spirit:** it still keeps state in memory during a run and knows nothing about HTTP, users or Postgres. The server is an application built on top of it, like `main.py`, and reuses `main.load_config()` and `main.build()`, so the UI runs exactly the agent the CLI runs.
- **One addition to the harness: run events** (`events.py`, a first slice of Phase 5). `Runtime.start(..., on_event=callback)` calls `callback(Event)` for `run.started`, `step.started`, `llm.response`, `tool.started`, `tool.completed`, `tool.offloaded`, `context.reduced`, `context.compacted` and `run.finished`. Without a callback nothing changes, and a callback that raises is logged and ignored.

## Data model

```
users ─┐                           ┌─ messages    every harness Message, in order (the agent's memory)
       ├─ sessions (owner_type, ───┤
incidents ┘   owner_id, state)     └─ runs ── run_events   one question + everything that happened
```

| Table | What it holds |
|---|---|
| `users`, `incidents` | Who or what a chat belongs to. |
| `sessions` | One chat. `owner_type` is `user` or `incident`, and `owner_id` is that row's id. Generated columns `user_id`/`incident_id` copy `owner_id` into the matching typed column and carry the real foreign keys, so a session can't point at a missing owner, and deleting an owner deletes its chats. `state` holds the `AgentState` minus messages: execution id, step count, token usage, and `extensions`, which include the compaction checkpoint and archive index. |
| `messages` | The full transcript. `body` is the harness `Message` exactly as produced (lossless); `role`/`text` are copies for display and SQL. Never edited or deleted, even after compaction. |
| `runs` | One user message and its outcome: status, answer, error, usage, steps. |
| `run_events` | The run's events, so reopening an old chat shows the same step-by-step activity. |

To add another kind of owner (for example a `service` or `ticket`): create its table, add the value to the `owner_type` CHECK, and add one more generated column with a foreign key.

## What happens when you send a message

1. `POST /api/sessions/{id}/messages` → `RunManager.start()`:
   - refuses (409) if this chat is already answering;
   - **loads the AgentState** from Postgres: `sessions.state` plus every row of `messages`. A new chat has none;
   - calls `runtime.start(agent, text, state=state, on_event=live.publish_event)` and records a `runs` row;
   - returns `{run_id}` immediately.
2. The browser opens `GET /api/runs/{run_id}/events` (server-sent events). The server first replays every event so far (so a late or reconnecting browser misses nothing), then sends new ones as the harness emits them.
3. The UI turns events into the timeline: steps, the model's notes, tool cards (click for arguments and the full result), an orange **artifact** badge when a result was offloaded, a purple notice when memory was compacted.
4. When the run ends, `save_run()` writes everything **in one transaction**: the new messages, the updated `sessions.state`, the run's outcome and its events. Then the server sends `run.persisted` and closes the stream, and the UI reloads the chat from the database.
5. The next message repeats step 1 with the saved state, so the agent continues with its whole history, including the compaction summary and archives from earlier turns.

**Stop** calls `POST /api/runs/{id}/cancel` → `Execution.cancel()`. The run ends as `cancelled`, unanswered tool calls get synthetic results (transcript repair), and what happened so far is saved.

**If the server restarts mid-run,** that run can't finish, so on the next startup it is marked `failed` (`SERVER_RESTARTED`). The chat's saved state is still the one from the previous completed run, so the chat stays usable.

## Memory panel

Shows what the agent carries between turns: steps and tokens so far, the context budget and compaction trigger, every compaction archive (click to read it, with a find box), and the current summary the model sees in place of older messages. **View full transcript** lists every stored message; the ones covered by the summary are faded, because the model no longer sees them directly.

## Built to grow into subagents, multi-agent and HITL

- Every event carries `execution_id` and `agent_id`. A subagent run would emit its own events with its own ids (plus a parent id), and the timeline can nest them under the tool call that started it.
- The timeline maps each event type to a renderer (`ui/src/components/Timeline.tsx`). A new type such as `subagent.started` or `approval.required` gets a renderer of its own. Until then, unknown types still appear as a generic row and are never hidden.
- HITL needs the browser to answer the server mid-run. That will be a `POST /api/runs/{id}/approvals/{call_id}` next to the existing cancel endpoint, with the harness pausing on an `ask()` decision from a `before_tool` hook. Neither part is built yet: today `ask()` still ends the tool call with `TOOL_APPROVAL_REQUIRED`.

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health`, `GET /api/config` | Server and DB check; model, tools and limits. |
| `GET/POST /api/users`, `GET/POST /api/incidents` | List or create owners. |
| `GET /api/sessions?owner_type=&owner_id=` | An owner's chats, newest first. |
| `POST /api/sessions` | New chat: `{owner_type, owner_id, title?}`. |
| `GET /api/sessions/{id}` | Chat with runs, their events, `active_run_id` and `memory`. |
| `GET /api/sessions/{id}/messages` | The raw transcript. |
| `PATCH /api/sessions/{id}`, `DELETE /api/sessions/{id}` | Rename or delete a chat (with everything in it). |
| `POST /api/sessions/{id}/messages` | Send a message, which starts a run: `{content}` → `{run_id}`. |
| `GET /api/runs/{id}/events` | Live events (SSE) while the run is live. |
| `POST /api/runs/{id}/cancel` | Stop a run. |
| `GET /api/artifacts/{id}` | An offloaded result or compaction archive, as text. |

FastAPI also serves interactive docs at http://localhost:8000/docs.

## Not included yet

- **Authentication.** Anyone who can open the UI can open any chat; users are records, not logins.
- **Artifacts stay on disk** (`.artifacts/`), not in Postgres; the database stores their ids.
- **Live events are per server process.** With several server processes, a browser must reach the one running its chat. A single process is assumed.
