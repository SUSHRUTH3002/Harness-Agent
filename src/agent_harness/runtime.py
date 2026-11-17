"""Runtime: owns the lifecycle of one agent execution.

initialize → prepare (registry, tool setup) → agent loop → completion → cleanup

`run()`/`start()` never raise for agent-level failures; they are reported in
the returned `RunResult`. They raise only for caller errors such as invalid
input.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Sequence

from pydantic import BaseModel

from agent_harness.agent import Agent
from agent_harness.cancellation import CancellationToken
from agent_harness.compaction import Compactor
from agent_harness.context import ContextManager, DefaultContextManager
from agent_harness.errors import ErrorCode, ErrorInfo, HarnessError, ValidationError
from agent_harness.events import EventBus, EventType, ObserverLike, RunEmitter, bind_emitter, unbind_emitter
from agent_harness.executor import SequentialToolExecutor, ToolExecutor
from agent_harness.hitl import Answer, Interaction, pending_interactions, submit_answers, take_held
from agent_harness.hooks import Hooks
from agent_harness.llm import LLMProvider, Usage
from agent_harness.loop import AgentLoop, LoopDependencies
from agent_harness.messages import Message, short_id, utcnow
from agent_harness.registry import ToolRegistry
from agent_harness.results import LargeResultProcessor
from agent_harness.state import AgentState, RunStatus
from agent_harness.tools import ToolResult

logger = logging.getLogger(__name__)

RunInput = str | Message | Sequence[Message]


class RunResult(BaseModel):
    execution_id: str
    agent_id: str
    status: RunStatus
    output: str | None
    error: ErrorInfo | None
    usage: Usage
    steps: int
    duration: float
    state: AgentState

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED

    @property
    def awaiting_input(self) -> bool:
        return self.status is RunStatus.AWAITING_INPUT

    @property
    def pending_interactions(self) -> list[Interaction]:
        """What the person is being asked (empty unless the run is AWAITING_INPUT)."""
        return pending_interactions(self.state) if self.awaiting_input else []


def _repair_transcript(state: AgentState, code: str, message: str) -> None:
    """Give every still-pending tool call a synthetic result.

    Called whenever a run ends abnormally (cancelled, timed out, or failed)
    partway through a tool batch, so the saved transcript always has exactly
    one result per call and stays valid to resend to any provider. Calls that had
    already finished in a suspended batch (hitl.py) keep their real results.
    """
    held = take_held(state)
    hitl = state.extensions.get("hitl")
    if hitl:
        hitl["pending"], hitl["answers"] = [], {}
    for call in state.pending_tool_calls():
        result = held.get(call.id) or ToolResult.error(call.id, call.name, message, code)
        state.append(Message.tool(result.call_id, result.content, name=result.name, is_error=result.is_error))
        state.tool_results.append(result)


class Execution:
    """A running (or finished) agent run, with the ability to cancel it.

    Cancellation is cooperative: `cancel()` both sets the run's
    `CancellationToken` (checked between loop steps, and exposed to tools via
    `ToolContext.cancel` for code that can't be interrupted by task
    cancellation, e.g. a sync tool body in a worker thread) and cancels the
    underlying asyncio task (interrupting whatever is currently awaited, such
    as an in-flight LLM or tool call). Either path ends the run as
    `RunStatus.CANCELLED`, never with a raised `CancelledError` -- see
    `Runtime._execute`.
    """

    def __init__(self, task: asyncio.Task[RunResult], cancel_token: CancellationToken, execution_id: str) -> None:
        self._task = task
        self._cancel_token = cancel_token
        self.execution_id = execution_id

    def cancel(self, reason: str | None = None) -> None:
        self._cancel_token.cancel(reason)
        self._task.cancel()

    @property
    def done(self) -> bool:
        return self._task.done()

    async def wait(self) -> RunResult:
        return await self._task


class Runtime:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        context_manager: ContextManager | None = None,
        tool_executor: ToolExecutor | None = None,
        loop: AgentLoop | None = None,
        hooks: Hooks | None = None,
        result_processor: LargeResultProcessor | None = None,
        compactor: Compactor | None = None,
        events: EventBus | None = None,
    ) -> None:
        if not isinstance(llm, LLMProvider):
            raise TypeError("llm must implement LLMProvider (an async generate(request) method)")
        self.llm = llm
        self.context_manager = context_manager or DefaultContextManager()
        # Only auto-wired into the default executor: a caller-supplied tool_executor manages
        # its own hooks wiring (or doesn't use these hooks at all), same as everywhere else
        # in this file where bringing your own component means owning its composition.
        self.hooks = hooks or Hooks()
        self.tool_executor = tool_executor or SequentialToolExecutor(hooks=self.hooks)
        self.loop = loop or AgentLoop()
        # None by default: large-result offloading is opt-in (Phase 7), same as hooks.
        self.result_processor = result_processor
        # None by default: compaction is opt-in (Phase 10) and also needs agent.limits.max_context_tokens.
        self.compactor = compactor
        # Observers for every run on this runtime (Phase 5). Per-run observers go to start(observers=...).
        self.events = events or EventBus()

    def start(
        self,
        agent: Agent,
        input: RunInput | None = None,
        *,
        state: AgentState | None = None,
        observers: Sequence[ObserverLike] = (),
        parent_execution_id: str | None = None,
        answers: Iterable[Answer] = (),
    ) -> Execution:
        """Start a run without waiting for it, returning a cancellable handle.

        `observers` receive this run's events only (on top of `self.events`, which sees every run).
        `answers` respond to a suspended run's pending interactions (see `resume`).
        """
        resumed = state is not None
        state = self._prepare_state(agent, input, state, list(answers))
        cancel_token = CancellationToken()
        emitter = RunEmitter(
            execution_id=state.execution_id, agent_id=state.agent_id, bus=self.events, observers=observers,
            state=state, parent_execution_id=parent_execution_id,
        )
        task = asyncio.ensure_future(self._execute(agent, state, cancel_token, emitter, resumed=resumed))
        return Execution(task, cancel_token, state.execution_id)

    async def run(
        self,
        agent: Agent,
        input: RunInput | None = None,
        *,
        state: AgentState | None = None,
        observers: Sequence[ObserverLike] = (),
        answers: Iterable[Answer] = (),
    ) -> RunResult:
        return await self.start(agent, input, state=state, observers=observers, answers=answers).wait()

    async def resume(
        self,
        agent: Agent,
        state: AgentState,
        answers: Iterable[Answer],
        *,
        observers: Sequence[ObserverLike] = (),
    ) -> RunResult:
        """Continue a run that ended AWAITING_INPUT, with the person's answers.

        `state` is usually loaded back from storage: `AgentState.model_validate_json(saved)`.
        Unanswered interactions stay pending (the run suspends again until they are answered).
        """
        if state.status is not RunStatus.AWAITING_INPUT:
            raise ValidationError(f"state is '{state.status.value}', not awaiting input", code=ErrorCode.INVALID_CONFIG)
        return await self.run(agent, state=state, observers=observers, answers=answers)

    def run_sync(
        self,
        agent: Agent,
        input: RunInput | None = None,
        *,
        state: AgentState | None = None,
        observers: Sequence[ObserverLike] = (),
    ) -> RunResult:
        return asyncio.run(self.run(agent, input, state=state, observers=observers))

    async def _execute(
        self, agent: Agent, state: AgentState, cancel_token: CancellationToken, emitter: RunEmitter, *,
        resumed: bool,
    ) -> RunResult:
        # Everything this task runs (loop, executor, tools, retry wrapper) emits through `emitter`.
        token = bind_emitter(emitter)
        try:
            return await self._execute_bound(agent, state, cancel_token, emitter, resumed=resumed)
        finally:
            unbind_emitter(token)

    async def _execute_bound(
        self, agent: Agent, state: AgentState, cancel_token: CancellationToken, emitter: RunEmitter, *,
        resumed: bool,
    ) -> RunResult:
        registry = ToolRegistry(agent.tools)
        deps = LoopDependencies(
            llm=self.llm, context=self.context_manager, executor=self.tool_executor, registry=registry,
            cancel=cancel_token, hooks=self.hooks, result_processor=self.result_processor,
            compactor=self.compactor,
        )
        exec_id = short_id(state.execution_id)
        logger.info(
            "exec=%s run started: agent=%s model=%s tools=%d history=%d messages%s",
            exec_id, agent.id, agent.model or "<provider default>", len(registry), len(state.messages),
            " (continuing existing state)" if resumed else "",
        )
        steps_before = state.step
        started = time.monotonic()
        state.started_at = state.started_at or utcnow()
        try:
            await emitter.emit(EventType.RUN_STARTED, {
                "agent": agent.id, "model": agent.model, "tools": registry.names(),
                "history_messages": len(state.messages), "resumed": resumed,
            })
            await registry.setup()
            logger.debug("exec=%s tools ready: %s", exec_id, ", ".join(registry.names()) or "none")
            try:
                if agent.limits.max_execution_time is not None:
                    async with asyncio.timeout(agent.limits.max_execution_time):
                        await self.loop.run(agent, state, deps)
                else:
                    await self.loop.run(agent, state, deps)
            finally:
                await registry.teardown()
        except asyncio.CancelledError:
            logger.warning("exec=%s run cancelled at step %d%s", exec_id, state.step,
                            f": {cancel_token.reason}" if cancel_token.reason else "")
            state.status = RunStatus.CANCELLED
            state.error = ErrorInfo(code=ErrorCode.CANCELLED, message=cancel_token.reason or "run was cancelled",
                                     type="CancelledError")
            _repair_transcript(state, ErrorCode.CANCELLED, "Run was cancelled before this tool call completed.")
        except TimeoutError:
            logger.warning("exec=%s run timed out at step %d after %.0fs (max_execution_time)",
                            exec_id, state.step, agent.limits.max_execution_time)
            state.status = RunStatus.TIMED_OUT
            state.error = ErrorInfo(
                code=ErrorCode.TIMEOUT,
                message=f"Run exceeded max_execution_time ({agent.limits.max_execution_time:.0f}s).",
                type="TimedOut",
            )
            _repair_transcript(state, ErrorCode.TIMEOUT, "Run timed out before this tool call completed.")
        except HarnessError as exc:
            logger.error("exec=%s run failed at step %d: [%s] %s", exec_id, state.step, exc.code, exc.message)
            state.status = RunStatus.FAILED
            state.error = ErrorInfo.from_exception(exc)
            _repair_transcript(state, exc.code, "Run failed before this tool call completed.")
        except Exception as exc:
            logger.exception("exec=%s run failed at step %d with an unexpected error", exec_id, state.step)
            state.status = RunStatus.FAILED
            state.error = ErrorInfo.from_exception(exc)
            _repair_transcript(state, ErrorCode.EXECUTION_ERROR, "Run failed before this tool call completed.")
        state.finished_at = utcnow()
        duration = time.monotonic() - started

        log = logger.info if state.status in (RunStatus.COMPLETED, RunStatus.AWAITING_INPUT) else logger.warning
        log(
            "exec=%s run finished: status=%s steps=%d duration=%.2fs tokens in=%d out=%d%s",
            exec_id, state.status.value, state.step - steps_before, duration,
            state.usage.input_tokens, state.usage.output_tokens,
            f" error={state.error.code}" if state.error else "",
        )
        try:
            await emitter.emit(EventType.RUN_COMPLETED, {
                "status": state.status.value, "steps": state.step - steps_before, "duration": round(duration, 3),
                "input_tokens": state.usage.input_tokens, "output_tokens": state.usage.output_tokens,
                "output_chars": len(state.result or ""),
                "pending_interactions": [i.id for i in pending_interactions(state)]
                if state.status is RunStatus.AWAITING_INPUT else [],
                "error": state.error.model_dump(include={"code", "message", "type"}) if state.error else None,
            })
        except asyncio.CancelledError:
            pass  # cancelled again while reporting; the result below is still correct
        return RunResult(
            execution_id=state.execution_id,
            agent_id=state.agent_id,
            status=state.status,
            output=state.result,
            error=state.error,
            usage=state.usage,
            steps=state.step - steps_before,
            duration=duration,
            state=state,
        )

    @staticmethod
    def _normalize_input(input: RunInput | None) -> list[Message]:
        if input is None:
            return []
        if isinstance(input, str):
            return [Message.user(input)]
        if isinstance(input, Message):
            return [input]
        messages = list(input)
        if not all(isinstance(m, Message) for m in messages):
            raise ValidationError("input must be a str, a Message, or a sequence of Messages", code=ErrorCode.INVALID_CONFIG)
        return messages

    def _prepare_state(
        self, agent: Agent, input: RunInput | None, state: AgentState | None, answers: list[Answer]
    ) -> AgentState:
        messages = self._normalize_input(input)
        if state is None:
            if answers:
                raise ValidationError("answers need the suspended run's state", code=ErrorCode.INVALID_CONFIG)
            if not messages:
                raise ValidationError("input is required to start a new run", code=ErrorCode.INVALID_CONFIG)
            state = AgentState(agent_id=agent.id)
        else:
            if state.agent_id != agent.id:
                raise ValidationError(
                    f"state belongs to agent '{state.agent_id}', not '{agent.id}'", code=ErrorCode.INVALID_CONFIG
                )
            if messages and state.pending_tool_calls():
                waiting = state.status is RunStatus.AWAITING_INPUT
                raise ValidationError(
                    "the run is waiting for input; answer its pending interactions with resume() first"
                    if waiting else "state has unresolved tool calls; resume it without new input first",
                    code=ErrorCode.INVALID_CONFIG,
                )
            if answers:
                submit_answers(state, answers)
            # Continuing a conversation or resuming: clear the previous run's outcome.
            _reset_outcome(state)
        for m in messages:
            state.append(m)
        return state


def _reset_outcome(state: AgentState) -> None:
    state.status = RunStatus.PENDING
    state.result = None
    state.error = None
    state.finished_at = None
    state.consecutive_truncations = 0
    state.partial_output = ""
