"""`RunBroadcast`: one run's events, readable by any number of clients, with replay.

`Execution.events()` has one reader and forgets what it has delivered. A chat UI needs more: a
second browser tab, a reconnect after a dropped connection, a client that opens the page while
the answer is still being written. A broadcast is attached to the run as an observer and keeps
its events; each subscriber reads from its own position (an event `seq`), so a client asks for
"everything after seq N" and gets the missed part and then the live part.

Durable events are always kept for the life of the broadcast. Transient ones (`llm.token`, ...)
are capped: when a client falls behind far enough that some were dropped, it is told with a
`Gap`, and reloads the chat (the finished answer is always in the saved messages).
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from dataclasses import dataclass
from typing import AsyncIterator

from agent_harness.errors import ErrorInfo
from agent_harness.events import Event
from agent_harness.state import RunStatus


@dataclass(frozen=True)
class Gap:
    """Streamed pieces the subscriber missed (dropped before it read them)."""


class RunBroadcast:
    def __init__(self, chat_id: str, *, max_transient: int = 5000) -> None:
        self.chat_id = chat_id
        self.max_transient = max_transient
        self._events: list[Event] = []
        self._transient = 0
        self._last_dropped_seq = 0
        self._waiters: set[asyncio.Event] = set()
        # Set by the first event (run.started is emitted once the chat is leased and saved).
        self.started = asyncio.Event()
        self.finished = False
        self.status: RunStatus | None = None
        self.error: ErrorInfo | None = None
        # The exception that stopped the run from starting at all (chat busy, bad answers, ...).
        self.start_failure: BaseException | None = None

    @property
    def dropped(self) -> bool:
        return self._last_dropped_seq > 0

    async def on_event(self, event: Event) -> None:
        self._events.append(event)
        if event.transient:
            self._transient += 1
            if self._transient > self.max_transient:
                self._drop_oldest_transient()
        self.started.set()
        self._wake()

    def _drop_oldest_transient(self) -> None:
        for i, event in enumerate(self._events):
            if event.transient:
                del self._events[i]
                self._transient -= 1
                self._last_dropped_seq = max(self._last_dropped_seq, event.seq)
                return

    def finish(self, status: RunStatus | None = None, error: ErrorInfo | None = None) -> None:
        self.finished = True
        self.status = status
        self.error = error
        self.started.set()
        self._wake()

    def _wake(self) -> None:
        for waiter in self._waiters:
            waiter.set()

    async def subscribe(self, after_seq: int = 0) -> AsyncIterator[Event | Gap]:
        """Events with `seq > after_seq`, then live ones; ends when the run has ended and all were read."""
        wakeup = asyncio.Event()
        self._waiters.add(wakeup)
        last, reported_gap = after_seq, 0
        try:
            while True:
                wakeup.clear()
                if self._last_dropped_seq > max(last, reported_gap):
                    reported_gap = self._last_dropped_seq
                    yield Gap()
                start = bisect_right(self._events, last, key=lambda e: e.seq)
                batch = self._events[start:]
                for event in batch:
                    last = event.seq
                    yield event
                if batch:
                    continue
                if self.finished:
                    return
                await wakeup.wait()
        finally:
            self._waiters.discard(wakeup)
