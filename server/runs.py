"""Live runs: start a harness run for a chat message, fan its events out to browsers, save it.

    POST message ──► RunManager.start()
                       ├─ load AgentState from Postgres (None for a new chat)
                       ├─ runtime.start(agent, text, state, on_event=live.publish)
                       └─ background: wait for the run ──► db.save_run() ──► "run.persisted"

    GET events (SSE) ─► live.subscribe(): everything published so far, then each new event

One run per session at a time. Live runs are kept in memory only while running
(plus a short grace period); after that, the events are read back from Postgres.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from agent_harness import Agent, Event, Execution, Runtime
from agent_harness.messages import utcnow

from server.db import Database

logger = logging.getLogger(__name__)

# How long a finished run stays in memory so a slow browser can still catch its last events.
_FINISHED_GRACE_SECONDS = 60


class RunConflict(Exception):
    """The session already has a run in progress."""


class LiveRun:
    def __init__(self, run_id: str, session_id: str) -> None:
        self.id = run_id
        self.session_id = session_id
        self.execution: Execution | None = None
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()

    def publish_event(self, event: Event) -> None:
        payload = event.model_dump(mode="json")
        self.publish(payload["type"], payload["step"], payload["data"], payload["at"])

    def publish(self, type: str, step: int, data: dict[str, Any], at: str | None = None) -> None:
        record = {"seq": len(self.events), "type": type, "step": step, "data": data,
                  "at": at or utcnow().isoformat(), "run_id": self.id}
        self.events.append(record)
        for queue in self._subscribers:
            queue.put_nowait(record)

    def close(self) -> None:
        self.done = True
        for queue in self._subscribers:
            queue.put_nowait(None)

    def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        """A queue pre-filled with every event so far, then fed live; None marks the end."""
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        for record in self.events:
            queue.put_nowait(record)
        if self.done:
            queue.put_nowait(None)
        else:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._subscribers.discard(queue)


class RunManager:
    def __init__(self, db: Database, runtime: Runtime, agent: Agent) -> None:
        self.db = db
        self.runtime = runtime
        self.agent = agent
        self.runs: dict[str, LiveRun] = {}
        self._active: dict[str, LiveRun] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def active_run(self, session_id: str) -> LiveRun | None:
        return self._active.get(session_id)

    async def start(self, session_id: str, text: str) -> LiveRun:
        if session_id in self._active:
            raise RunConflict("This chat is still answering the previous message.")
        live = LiveRun(str(uuid.uuid4()), session_id)
        self._active[session_id] = live  # claimed before any await, so two quick sends can't both start
        try:
            state = await self.db.load_state(session_id)
            first_new = len(state.messages) if state else 0
            live.execution = self.runtime.start(self.agent, text, state=state, on_event=live.publish_event)
            await self.db.create_run(live.id, session_id, text)
        except BaseException:
            del self._active[session_id]
            if live.execution is not None:
                live.execution.cancel("could not record the run")
            raise
        self.runs[live.id] = live
        task = asyncio.create_task(self._finish(live, first_new))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info("run %s started for session %s", live.id[:8], session_id[:8])
        return live

    def cancel(self, run_id: str) -> bool:
        live = self.runs.get(run_id)
        if live is None or live.done or live.execution is None:
            return False
        live.execution.cancel("cancelled from the UI")
        return True

    async def _finish(self, live: LiveRun, first_new: int) -> None:
        try:
            result = await live.execution.wait()
            await self.db.save_run(live.id, live.session_id, result, first_new, live.events)
            live.publish("run.persisted", result.state.step, {"status": result.status.value})
        except Exception as exc:
            logger.exception("run %s: saving to the database failed", live.id[:8])
            live.publish("run.persist_failed", 0, {"message": f"{type(exc).__name__}: {exc}"})
            try:
                await self.db.fail_run(live.id, str(exc))
            except Exception:
                logger.exception("run %s: could not mark the run as failed", live.id[:8])
        finally:
            self._active.pop(live.session_id, None)
            live.close()
            asyncio.get_running_loop().call_later(_FINISHED_GRACE_SECONDS, self.runs.pop, live.id, None)

    async def shutdown(self) -> None:
        for live in list(self._active.values()):
            if live.execution is not None:
                live.execution.cancel("server shutting down")
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=10)
