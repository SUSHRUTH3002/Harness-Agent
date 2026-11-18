"""LLM-call reliability middleware: retry with backoff, and a per-attempt timeout.

`ResilientLLMProvider` wraps any `LLMProvider`, so retry/timeout policy stays
outside the agent loop (see design.md D10) -- the loop calls `generate()`
exactly as it would on the inner provider.

It streams too (Phase 8): `stream()` retries a stream that fails, even partway through. The
pieces already passed on are then withdrawn with a `reset` piece (announced as `llm.stream_reset`),
so whoever shows them clears the half answer before the retry streams it again.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from typing import AsyncIterator, Awaitable, Callable

from pydantic import BaseModel, Field

from agent_harness.errors import ErrorCode, LLMError
from agent_harness.events import EventType, emit_event
from agent_harness.llm import LLMProvider, LLMRequest, LLMResponse, LLMStreamEvent
from agent_harness.streaming import open_stream, timed_stream

logger = logging.getLogger(__name__)

# Codes worth retrying: transient or provider-side. AUTH, INVALID_REQUEST and
# CONTEXT_WINDOW_EXCEEDED are not included -- retrying an unchanged request
# would fail identically.
DEFAULT_RETRYABLE_CODES = frozenset(
    {ErrorCode.RATE_LIMIT, ErrorCode.SERVER, ErrorCode.TIMEOUT, ErrorCode.TRANSPORT, ErrorCode.EMPTY_RESPONSE}
)


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=3, ge=0)
    initial_backoff: float = Field(default=0.5, gt=0)
    max_backoff: float = Field(default=20.0, gt=0)
    multiplier: float = Field(default=2.0, gt=1)
    # Randomizes each delay by +/- this fraction, so many concurrent callers don't retry in lockstep.
    jitter: float = Field(default=0.2, ge=0, le=1)
    retryable_codes: frozenset[str] = Field(default_factory=lambda: DEFAULT_RETRYABLE_CODES)

    def should_retry(self, error: LLMError, attempt: int) -> bool:
        return attempt < self.max_retries and error.code in self.retryable_codes

    def delay_for(self, error: LLMError, attempt: int) -> float:
        retry_after = error.details.get("retry_after")
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return min(float(retry_after), self.max_backoff)
        base = min(self.max_backoff, self.initial_backoff * (self.multiplier**attempt))
        return base * (1 + random.uniform(-self.jitter, self.jitter))


class ResilientLLMProvider:
    """Wraps an `LLMProvider` with a per-attempt timeout and retry-with-backoff.

    `timeout` bounds each individual attempt; a timed-out attempt counts as a
    retryable `LLMError(code=TIMEOUT)`. `idle_timeout` (streaming only) bounds the
    gap between two pieces once the answer has started; a stall is retried the same way.
    `sleep` is overridable so a caller can substitute a faster clock without waiting
    out real backoff delays.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        policy: RetryPolicy | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._llm = llm
        self.policy = policy or RetryPolicy()
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self._sleep = sleep

    async def generate(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                if self.timeout is not None:
                    async with asyncio.timeout(self.timeout):
                        return await self._llm.generate(request)
                return await self._llm.generate(request)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                error = LLMError(
                    f"LLM call timed out after {self.timeout:.0f}s", code=ErrorCode.TIMEOUT, retryable=True
                )
            except LLMError as exc:
                error = exc

            attempt = await self._before_retry(error, attempt, started)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Stream with the same retry policy. A retry after pieces were passed on yields `reset` first."""
        attempt = 0
        while True:
            started = time.perf_counter()
            passed_on = 0
            try:
                # aclosing: if our consumer stops early (cancel), the model connection closes now.
                async with contextlib.aclosing(timed_stream(open_stream(self._llm, request), total=self.timeout,
                                                            idle=self.idle_timeout)) as pieces:
                    async for event in pieces:
                        passed_on += 1
                        yield event
                return
            except asyncio.CancelledError:
                raise
            except LLMError as exc:
                error = exc
            attempt = await self._before_retry(error, attempt, started)
            if passed_on:
                yield LLMStreamEvent(kind="reset", attempt=attempt)

    async def _before_retry(self, error: LLMError, attempt: int, started: float) -> int:
        """Raise `error` if it should not be retried; otherwise announce the retry, wait, and
        return the next attempt number."""
        elapsed = time.perf_counter() - started
        if not self.policy.should_retry(error, attempt):
            error.details.setdefault("attempts", attempt + 1)
            logger.warning(
                "llm call failed permanently after %d attempt(s), %.2fs: [%s] %s",
                attempt + 1, elapsed, error.code, error.message,
            )
            raise error
        delay = self.policy.delay_for(error, attempt)
        attempt += 1
        logger.warning(
            "llm call failed (attempt %d/%d, %.2fs): [%s] %s -- retrying in %.1fs",
            attempt, self.policy.max_retries, elapsed, error.code, error.message, delay,
        )
        await emit_event(EventType.LLM_RETRYING, {
            "attempt": attempt, "max_retries": self.policy.max_retries, "code": error.code,
            "message": error.message, "delay": round(delay, 2), "elapsed": round(elapsed, 3),
        })
        await self._sleep(delay)
        return attempt
