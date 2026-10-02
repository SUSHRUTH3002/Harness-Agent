"""Runtime: owns the lifecycle of one agent execution.

initialize → prepare (registry, tool setup) → agent loop → completion → cleanup

`run()` does not raise for agent-level failures; they are reported in the
returned `RunResult`. It raises only for caller errors such as invalid input.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from pydantic import BaseModel

from agent_harness.agent import Agent
from agent_harness.context import ContextManager, DefaultContextManager
from agent_harness.errors import ErrorCode, ErrorInfo, HarnessError, ValidationError
from agent_harness.executor import SequentialToolExecutor, ToolExecutor
from agent_harness.llm import LLMProvider, Usage
from agent_harness.loop import AgentLoop, LoopDependencies
from agent_harness.messages import Message, short_id, utcnow
from agent_harness.registry import ToolRegistry
from agent_harness.state import AgentState, RunStatus

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


class Runtime:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        context_manager: ContextManager | None = None,
        tool_executor: ToolExecutor | None = None,
        loop: AgentLoop | None = None,
    ) -> None:
        if not isinstance(llm, LLMProvider):
            raise TypeError("llm must implement LLMProvider (an async generate(request) method)")
        self.llm = llm
        self.context_manager = context_manager or DefaultContextManager()
        self.tool_executor = tool_executor or SequentialToolExecutor()
        self.loop = loop or AgentLoop()

    async def run(self, agent: Agent, input: RunInput | None = None, *, state: AgentState | None = None) -> RunResult:
        resumed = state is not None
        state = self._prepare_state(agent, input, state)
        registry = ToolRegistry(agent.tools)
        deps = LoopDependencies(
            llm=self.llm, context=self.context_manager, executor=self.tool_executor, registry=registry
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
            await registry.setup()
            logger.debug("exec=%s tools ready: %s", exec_id, ", ".join(registry.names()) or "none")
            try:
                await self.loop.run(agent, state, deps)
            finally:
                await registry.teardown()
        except HarnessError as exc:
            logger.error("exec=%s run failed at step %d: [%s] %s", exec_id, state.step, exc.code, exc.message)
            state.status = RunStatus.FAILED
            state.error = ErrorInfo.from_exception(exc)
        except Exception as exc:
            logger.exception("exec=%s run failed at step %d with an unexpected error", exec_id, state.step)
            state.status = RunStatus.FAILED
            state.error = ErrorInfo.from_exception(exc)
        state.finished_at = utcnow()
        duration = time.monotonic() - started

        log = logger.info if state.status is RunStatus.COMPLETED else logger.warning
        log(
            "exec=%s run finished: status=%s steps=%d duration=%.2fs tokens in=%d out=%d%s",
            exec_id, state.status.value, state.step - steps_before, duration,
            state.usage.input_tokens, state.usage.output_tokens,
            f" error={state.error.code}" if state.error else "",
        )
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

    def run_sync(self, agent: Agent, input: RunInput | None = None, *, state: AgentState | None = None) -> RunResult:
        return asyncio.run(self.run(agent, input, state=state))

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

    def _prepare_state(self, agent: Agent, input: RunInput | None, state: AgentState | None) -> AgentState:
        messages = self._normalize_input(input)
        if state is None:
            if not messages:
                raise ValidationError("input is required to start a new run", code=ErrorCode.INVALID_CONFIG)
            state = AgentState(agent_id=agent.id)
        else:
            if state.agent_id != agent.id:
                raise ValidationError(
                    f"state belongs to agent '{state.agent_id}', not '{agent.id}'", code=ErrorCode.INVALID_CONFIG
                )
            if messages and state.pending_tool_calls():
                raise ValidationError(
                    "state has unresolved tool calls; resume it without new input first",
                    code=ErrorCode.INVALID_CONFIG,
                )
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
