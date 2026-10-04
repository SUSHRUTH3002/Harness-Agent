"""Cooperative cancellation signal threaded through a run.

Awaited code (an LLM call, an async tool body) already receives
`asyncio.CancelledError` automatically when its task is cancelled. A
`CancellationToken` additionally lets code that cannot rely on that -- a sync
tool body running in a worker thread, or a loop between awaits -- check
cancellation cooperatively instead of being forcibly interrupted.
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self.reason = reason or self.reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise asyncio.CancelledError(self.reason or "cancelled")

    async def wait(self) -> None:
        await self._event.wait()
