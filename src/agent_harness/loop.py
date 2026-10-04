"""The agent loop: build context, call the LLM, run tools, repeat.

The loop knows nothing about specific tools, providers or applications.
Exceptions from the LLM propagate to the Runtime, which records them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from agent_harness.agent import Agent
from agent_harness.cancellation import CancellationToken
from agent_harness.context import ContextManager
from agent_harness.errors import ErrorCode, ErrorInfo, LLMError
from agent_harness.executor import ToolExecutionContext, ToolExecutor
from agent_harness.llm import FinishReason, LLMProvider, LLMRequest, LLMResponse
from agent_harness.messages import Message, ToolCall, preview, short_id
from agent_harness.registry import ToolRegistry
from agent_harness.state import AgentState, RunStatus
from agent_harness.tools import ToolResult

logger = logging.getLogger(__name__)

CONTINUE_PROMPT = (
    "Your previous response was cut off because it reached the maximum output length. "
    "Continue exactly where you left off, without repeating what you already wrote."
)
REISSUE_TOOL_CALL_PROMPT = (
    "Your previous response was cut off because it reached the maximum output length, "
    "so the incomplete tool call was discarded and not executed. Issue it again, keeping it shorter."
)


@dataclass
class LoopDependencies:
    llm: LLMProvider
    context: ContextManager
    executor: ToolExecutor
    registry: ToolRegistry
    cancel: CancellationToken = field(default_factory=CancellationToken)


def _tool_call_signature(calls: list[ToolCall]) -> frozenset[str]:
    """A step's tool calls, reduced to (name, canonical arguments) pairs for repeat detection."""
    return frozenset(
        f"{c.name}:{json.dumps(c.arguments, sort_keys=True) if c.arguments is not None else c.raw_arguments}"
        for c in calls
    )


class AgentLoop:
    async def run(self, agent: Agent, state: AgentState, deps: LoopDependencies) -> AgentState:
        exec_id = short_id(state.execution_id)
        state.status = RunStatus.RUNNING

        # Resuming a state that stopped between the LLM call and tool execution.
        pending = state.pending_tool_calls()
        if pending:
            logger.info("exec=%s resuming: executing %d pending tool call(s) first", exec_id, len(pending))
            await self._execute_tools(pending, state, deps, agent)

        steps_this_run = 0
        while True:
            deps.cancel.raise_if_cancelled()
            if steps_this_run >= agent.limits.max_steps:
                logger.warning("exec=%s stopped: max_steps=%d reached", exec_id, agent.limits.max_steps)
                state.status = RunStatus.MAX_STEPS
                return state
            steps_this_run += 1
            state.step += 1
            logger.info("exec=%s step=%d started (%d/%d this run)", exec_id, state.step, steps_this_run,
                        agent.limits.max_steps)

            request = await deps.context.build(agent, state, deps.registry.schemas())
            logger.debug("exec=%s step=%d calling LLM: %d messages, %d tools", exec_id, state.step,
                         len(request.messages), len(request.tools))
            started = time.perf_counter()

            response = await self._generate(agent, deps, request)

            state.usage = state.usage + response.usage
            calls = response.message.tool_calls

            logger.info(
                "exec=%s step=%d LLM responded in %.2fs: finish=%s tool_calls=%d tokens in=%d out=%d",
                exec_id, state.step, time.perf_counter() - started, response.finish_reason.value, len(calls),
                response.usage.input_tokens, response.usage.output_tokens,
            )
            if response.message.text:
                logger.info("exec=%s step=%d assistant: %s", exec_id, state.step, preview(response.message.text))

            if response.finish_reason is FinishReason.LENGTH:
                if self._continue_after_truncation(agent, state, response):
                    continue
                return state
            state.consecutive_truncations = 0

            state.append(response.message)
            if not calls:
                state.result = state.partial_output + response.message.text
                state.partial_output = ""
                state.status = RunStatus.COMPLETED
                logger.info("exec=%s step=%d final answer produced (%d chars)", exec_id, state.step,
                            len(state.result))
                return state

            state.partial_output = ""
            await self._execute_tools(calls, state, deps, agent)
            if self._check_repeated_calls(agent, state, calls):
                return state

    async def _generate(self, agent: Agent, deps: LoopDependencies, request: LLMRequest) -> LLMResponse:
        """Call the LLM, honoring `agent.limits.llm_timeout` if set.

        This is a hard per-agent cap, separate from any retry/timeout a provider
        is itself wrapped with (e.g. `ResilientLLMProvider`); it never retries.
        """
        if agent.limits.llm_timeout is None:
            return await deps.llm.generate(request)
        try:
            async with asyncio.timeout(agent.limits.llm_timeout):
                return await deps.llm.generate(request)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise LLMError(
                f"LLM call exceeded agent.limits.llm_timeout ({agent.limits.llm_timeout:.0f}s)",
                code=ErrorCode.TIMEOUT,
            ) from exc

    def _check_repeated_calls(self, agent: Agent, state: AgentState, calls: list[ToolCall]) -> bool:
        """Detect the model issuing the same tool call(s) repeatedly. Returns True if the run stopped.

        Tracked as a signature of the *previous* step's calls against this step's,
        stored in `state.extensions` (never a dedicated field -- see design.md D on extensions).
        """
        threshold = agent.limits.repeat_call_threshold
        if threshold is None:
            return False
        exec_id = short_id(state.execution_id)
        guard = state.extensions.setdefault("repeat_guard", {"signature": None, "count": 0})
        signature = sorted(_tool_call_signature(calls))
        if signature == guard["signature"]:
            guard["count"] += 1
        else:
            guard["signature"] = signature
            guard["count"] = 1
        if guard["count"] < threshold:
            return False

        names = ", ".join(sorted({c.name for c in calls}))
        if agent.limits.repeat_call_action == "stop":
            logger.error(
                "exec=%s step=%d stopped: %s called with identical arguments %d times in a row",
                exec_id, state.step, names, guard["count"],
            )
            state.status = RunStatus.FAILED
            state.error = ErrorInfo(
                code=ErrorCode.REPEATED_TOOL_CALL,
                message=f"'{names}' was called with identical arguments {guard['count']} times in a row.",
                type="RepeatedToolCall",
                details={"count": guard["count"], "tools": sorted({c.name for c in calls})},
            )
            return True

        logger.warning(
            "exec=%s step=%d %s called with identical arguments %d times in a row; reminding the model",
            exec_id, state.step, names, guard["count"],
        )
        state.append(
            Message.user(
                f"You have called {names} with the exact same arguments {guard['count']} times in a row. "
                "If you already have the information you need, use it to answer now instead of calling it again.",
                metadata={"source": "harness", "reason": "repeated_tool_call"},
            )
        )
        return False

    def _continue_after_truncation(self, agent: Agent, state: AgentState, response: LLMResponse) -> bool:
        """Record a cut-off response and decide whether to ask the model to continue.

        Incomplete tool calls are dropped: their arguments cannot be trusted.
        """
        exec_id = short_id(state.execution_id)
        message = response.message
        partial_text = message.text
        dropped_tool_call = bool(message.tool_calls)
        state.partial_output += partial_text
        if partial_text:
            state.append(
                Message.assistant(partial_text, id=message.id, metadata={**message.metadata, "truncated": True})
            )

        limit = agent.limits.max_truncation_continuations
        if state.consecutive_truncations >= limit:
            logger.error(
                "exec=%s step=%d output still truncated after %d continuation(s); giving up",
                exec_id, state.step, limit,
            )
            state.status = RunStatus.FAILED
            state.result = state.partial_output or None
            state.error = ErrorInfo(
                code=ErrorCode.OUTPUT_TRUNCATED,
                message=(
                    "Model output was cut off at the maximum output length "
                    f"after {limit} continuation attempt(s)."
                ),
                type="OutputTruncated",
                details={"continuations": limit},
            )
            return False

        state.consecutive_truncations += 1
        logger.warning(
            "exec=%s step=%d output truncated at max length%s; asking model to continue (%d/%d)",
            exec_id, state.step, " (incomplete tool call discarded)" if dropped_tool_call else "",
            state.consecutive_truncations, limit,
        )
        prompt = REISSUE_TOOL_CALL_PROMPT if dropped_tool_call else CONTINUE_PROMPT
        state.append(Message.user(prompt, metadata={"source": "harness", "reason": "output_truncated"}))
        return True

    async def _execute_tools(
        self, calls: list[ToolCall], state: AgentState, deps: LoopDependencies, agent: Agent
    ) -> None:
        exec_id = short_id(state.execution_id)
        logger.info("exec=%s step=%d executing %d tool call(s): %s", exec_id, state.step, len(calls),
                    ", ".join(c.name for c in calls))
        context = ToolExecutionContext(
            execution_id=state.execution_id,
            agent_id=state.agent_id,
            step=state.step,
            cancel=deps.cancel,
            default_tool_timeout=agent.limits.tool_timeout,
        )
        results = await deps.executor.execute(calls, deps.registry, context)

        # Enforce the executor contract so the transcript is always valid for providers.
        by_id = {r.call_id: r for r in results}
        for call in calls:
            result = by_id.get(call.id)
            if result is None:
                logger.error("exec=%s executor returned no result for call %s (%s)", exec_id, call.id, call.name)
                result = ToolResult.error(
                    call.id, call.name, "Tool executor produced no result for this call.", ErrorCode.EXECUTION_ERROR
                )
            state.append(Message.tool(result.call_id, result.content, name=result.name, is_error=result.is_error))
            state.tool_results.append(result)
        failed = sum(r.is_error for r in state.tool_results[-len(calls):])
        logger.info("exec=%s step=%d tools done: %d ok, %d failed", exec_id, state.step, len(calls) - failed, failed)
