"""Tool execution.

Contract for every `ToolExecutor`: exactly one `ToolResult` per call, in call
order, and no exception escapes for tool-level failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from agent_harness.cancellation import CancellationToken
from agent_harness.errors import ErrorCode, HarnessError, ValidationError
from agent_harness.events import EventType, emit_event
from agent_harness.hitl import (
    Answer,
    InputRequired,
    Interaction,
    awaiting_result,
    deferred_result,
    is_awaiting,
    rejected_result,
    with_note,
)
from agent_harness.hooks import Hooks
from agent_harness.messages import ToolCall, preview, short_id
from agent_harness.registry import ToolRegistry
from agent_harness.tools import Tool, ToolContext, ToolResult, normalize_output

logger = logging.getLogger(__name__)

_MAX_LISTED_TOOLS = 50


@dataclass(frozen=True)
class ToolExecutionContext:
    execution_id: str
    agent_id: str
    step: int
    cancel: CancellationToken = field(default_factory=CancellationToken)
    # Default timeout for tools that don't set their own `Tool.timeout` (agent.limits.tool_timeout).
    default_tool_timeout: float | None = None
    # Most tool bodies running at once within one batch (ParallelToolExecutor only).
    max_parallel_tool_calls: int = 10
    # Answers being applied on resume (hitl.py): {call_id: (interaction, answer)}.
    answers: dict[str, tuple[Interaction, Answer]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolExecutor(Protocol):
    async def execute(
        self, calls: Sequence[ToolCall], registry: ToolRegistry, context: ToolExecutionContext
    ) -> list[ToolResult]: ...


def parse_arguments(call: ToolCall) -> dict[str, Any]:
    if call.arguments is not None:
        return dict(call.arguments)
    raw = call.raw_arguments
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Arguments for tool '{call.name}' are not valid JSON: {exc.msg}", code=ErrorCode.INVALID_ARGS
        ) from exc
    if not isinstance(value, dict):
        raise ValidationError(f"Arguments for tool '{call.name}' must be a JSON object", code=ErrorCode.INVALID_ARGS)
    return value


@dataclass
class _PreparedCall:
    """A call that passed lookup, validation and `before_tool`: ready for its tool body to run."""

    tool: Tool
    arguments: dict[str, Any]
    # The person's approval/edit, when this call was approved on resume (its note goes to the model).
    approval: Answer | None = None


def _log_started(call: ToolCall, context: ToolExecutionContext) -> None:
    logger.info(
        "exec=%s step=%d TOOL %s started args=%s",
        short_id(context.execution_id), context.step, call.name,
        preview(json.dumps(call.arguments) if call.arguments is not None else str(call.raw_arguments)),
    )


async def _emit_started(call: ToolCall, prepared: _PreparedCall) -> None:
    """`tool.started`: the tool body is about to run (calls stopped by `_prepare` never start)."""
    await emit_event(EventType.TOOL_STARTED, {"call_id": call.id, "tool": call.name, "arguments": prepared.arguments})


async def _emit_completed(call: ToolCall, result: ToolResult, elapsed: float, ran: bool) -> None:
    """`tool.completed`: the final result, after `after_tool`. `ran` is False for a call
    rejected before its body started (unknown tool, invalid arguments, denied)."""
    await emit_event(EventType.TOOL_COMPLETED, {
        "call_id": call.id, "tool": call.name, "ok": not result.is_error, "error_code": result.error_code,
        "ran": ran, "duration": round(elapsed, 3), "content_chars": len(result.content),
        "content": preview(result.content, 500),
    })


async def _report_not_finished(call: ToolCall, result: ToolResult, context: ToolExecutionContext) -> None:
    """A call that is waiting for a person (or deferred behind one): no after_tool, no tool.completed."""
    exec_id = short_id(context.execution_id)
    if result.metadata.get("deferred"):
        logger.info("exec=%s step=%d tool %s deferred until pending input is answered", exec_id, context.step,
                    call.name)
        await emit_event(EventType.TOOL_DEFERRED, {"call_id": call.id, "tool": call.name})
        return
    interaction = result.metadata.get("interaction", {})
    logger.info("exec=%s step=%d tool %s awaiting %s: %s", exec_id, context.step, call.name,
                interaction.get("kind", "input"), preview(result.content))
    await emit_event(EventType.TOOL_AWAITING_INPUT, {
        "call_id": call.id, "tool": call.name, "interaction_id": interaction.get("id"), "kind": interaction.get("kind"),
    })


def _concurrency_safe(registry: ToolRegistry, call: ToolCall) -> bool:
    tool = registry.get(call.name)
    return tool is not None and tool.concurrency_safe


def _log_finished(call: ToolCall, result: ToolResult, context: ToolExecutionContext, elapsed: float) -> None:
    exec_id = short_id(context.execution_id)
    if result.is_error:
        logger.warning(
            "exec=%s step=%d tool %s failed in %.3fs [%s]: %s",
            exec_id, context.step, call.name, elapsed, result.error_code, preview(result.content),
        )
    else:
        logger.info(
            "exec=%s step=%d tool %s ok in %.3fs (%d chars)",
            exec_id, context.step, call.name, elapsed, len(result.content),
        )
        logger.debug("exec=%s step=%d tool %s result: %s", exec_id, context.step, call.name,
                     preview(result.content))


class SequentialToolExecutor:
    """Runs calls one at a time, in the order the model issued them.

    Each call goes through two stages, which `ParallelToolExecutor` reuses:
    `_prepare` (lookup, argument validation, `before_tool` policy) and
    `_invoke` (the tool body, with its timeout). `after_tool` runs last.
    """

    def __init__(self, hooks: Hooks | None = None) -> None:
        self.hooks = hooks or Hooks()

    async def execute(
        self, calls: Sequence[ToolCall], registry: ToolRegistry, context: ToolExecutionContext
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            # Once a call is waiting for a person, later non-concurrency-safe calls wait too:
            # they may depend on it. (They are still checked, so their own approvals are asked now.)
            hold = any(is_awaiting(r) for r in results) and not _concurrency_safe(registry, call)
            results.append(await self.execute_one(call, registry, context, hold=hold))
        return results

    async def execute_one(
        self, call: ToolCall, registry: ToolRegistry, context: ToolExecutionContext, *, hold: bool = False
    ) -> ToolResult:
        started = time.perf_counter()
        _log_started(call, context)
        prepared, ctx = await self._prepare(call, registry, context)
        ran = not isinstance(prepared, ToolResult)
        if ran and hold:
            result = deferred_result(call.id, call.name)
        elif ran:
            await _emit_started(call, prepared)
            result = await self._invoke(call, prepared, ctx, context)
        else:
            result = prepared
        if is_awaiting(result):
            await _report_not_finished(call, result, context)
            return result
        result = await self.hooks.run_after_tool(call, result, ctx)
        elapsed = time.perf_counter() - started
        _log_finished(call, result, context, elapsed)
        await _emit_completed(call, result, elapsed, ran)
        return result

    async def _prepare(
        self, call: ToolCall, registry: ToolRegistry, context: ToolExecutionContext
    ) -> tuple[ToolResult | _PreparedCall, ToolContext]:
        """Everything before the tool body: lookup, validation and the `before_tool` policy check.

        Returns either a final (error) result, or the call ready to run. Always returns the
        `ToolContext` too (even for a lookup/validation failure, using a fallback context), so
        `after_tool` sees the same object `before_tool` saw -- e.g. anything a `before_tool`
        hook stashed in `ctx.metadata` for a later `after_tool` hook to read is preserved.
        """
        ctx = ToolContext(
            execution_id=context.execution_id, agent_id=context.agent_id, call_id=call.id,
            tool_name=call.name, step=context.step, cancel=context.cancel, metadata=dict(context.metadata),
        )
        tool = registry.get(call.name)
        if tool is None:
            available = ", ".join(registry.names()[:_MAX_LISTED_TOOLS]) or "none"
            return ToolResult.error(
                call.id, call.name, f"Unknown tool '{call.name}'. Available tools: {available}", ErrorCode.UNKNOWN_TOOL
            ), ctx

        # Resuming after a person answered this call (hitl.py).
        interaction, answer = context.answers.get(call.id, (None, None))
        if answer is not None and answer.action == "reject":
            logger.info("exec=%s step=%d tool %s rejected by the user", short_id(context.execution_id),
                        context.step, call.name)
            return rejected_result(call.id, call.name, interaction.kind, answer.note), ctx
        try:
            raw = dict(answer.arguments) if answer is not None and answer.action == "edit" else parse_arguments(call)
            arguments = tool.validate_arguments(raw)
        except HarnessError as exc:
            message = f"The user's edited arguments are invalid: {exc.message}" if answer is not None else exc.message
            return ToolResult.error(call.id, call.name, message, exc.code), ctx
        if answer is not None and interaction.kind == "input":
            ctx.answer = answer  # the tool runs again, now with the person's answer
        if answer is not None and answer.action == "edit":
            call = call.model_copy(update={"arguments": arguments, "raw_arguments": None})

        decision = await self.hooks.run_before_tool(call, ctx)
        approved = answer is not None and interaction.kind == "approval"
        if decision.action == "ask" and not approved:
            message = decision.reason or f"Tool '{call.name}' requires approval before it runs."
            logger.warning("exec=%s step=%d tool %s needs approval: %s", short_id(context.execution_id),
                           context.step, call.name, message)
            return awaiting_result(Interaction(
                kind="approval", call_id=call.id, tool=call.name, arguments=arguments, message=message,
                allow_edit=decision.allow_edit, step=context.step,
            )), ctx
        if decision.action == "deny":
            message = decision.reason or f"Tool '{call.name}' was denied."
            logger.warning("exec=%s step=%d tool %s denied: %s", short_id(context.execution_id), context.step,
                           call.name, message)
            return ToolResult.error(call.id, call.name, message, ErrorCode.TOOL_DENIED), ctx
        return _PreparedCall(tool, arguments, approval=answer if approved else None), ctx

    async def _invoke(
        self, call: ToolCall, prepared: _PreparedCall, ctx: ToolContext, context: ToolExecutionContext
    ) -> ToolResult:
        """Run the tool body with its timeout. Never raises, except for cancellation."""
        result = await self._run_body(call, prepared, ctx, context)
        if prepared.approval is not None and not is_awaiting(result):
            result = with_note(result, prepared.approval)
        return result

    async def _run_body(
        self, call: ToolCall, prepared: _PreparedCall, ctx: ToolContext, context: ToolExecutionContext
    ) -> ToolResult:
        tool = prepared.tool
        timeout = tool.timeout if tool.timeout is not None else context.default_tool_timeout
        try:
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    value = await tool.execute(prepared.arguments, ctx)
            else:
                value = await tool.execute(prepared.arguments, ctx)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return ToolResult.error(
                call.id, call.name, f"Tool '{call.name}' timed out after {timeout:.0f}s", ErrorCode.TIMEOUT
            )
        except InputRequired as exc:
            return awaiting_result(Interaction(
                kind="input", call_id=call.id, tool=call.name, arguments=prepared.arguments, message=exc.message,
                options=exc.options, multi_select=exc.multi_select, allow_free_text=exc.allow_free_text,
                step=context.step, metadata=exc.metadata,
            ))
        except HarnessError as exc:
            return ToolResult.error(call.id, call.name, exc.message, exc.code)
        except Exception as exc:
            # Unexpected tool bug: keep the traceback in the logs, give the model a clean message.
            logger.debug("tool %s raised", call.name, exc_info=True)
            return ToolResult.error(
                call.id, call.name, f"Tool '{call.name}' failed: {type(exc).__name__}: {exc}", ErrorCode.TOOL_ERROR
            )

        try:
            return normalize_output(call.id, call.name, value)
        except Exception as exc:
            return ToolResult.error(
                call.id,
                call.name,
                f"Tool '{call.name}' returned a value that could not be serialized: {type(exc).__name__}",
                ErrorCode.TOOL_ERROR,
            )


class ParallelToolExecutor(SequentialToolExecutor):
    """Runs `concurrency_safe` tools at the same time; everything else keeps sequential semantics.

    For one batch of calls (one model response):

    1. **Prepare, in model order:** lookup, validation and `before_tool` run one call at a
       time, before any tool body starts -- so policy (and, later, approval) is never asked
       about two calls at once, and a denied call simply already has its result.
    2. **Run:** consecutive `concurrency_safe` calls form a group and run together, at most
       `context.max_parallel_tool_calls` at once (a sliding pool: as one finishes, the next
       starts). A call that is *not* concurrency-safe is a barrier: it waits for everything
       before it, runs alone, and everything after it waits for it -- so reads issued after
       a write see the write.
    3. **Finish, in model order:** `after_tool` runs for each call, and results are returned
       in the order the model issued them, however they finished.

    One call failing or timing out only affects that call's result. Cancelling the run
    cancels every call still running.

    `max_total`: optional cap on tool bodies running at once across *all* runs that share
    this executor (e.g. many chats on one server), on top of the per-batch cap.
    """

    def __init__(self, hooks: Hooks | None = None, *, max_total: int | None = None) -> None:
        super().__init__(hooks)
        self._total = asyncio.Semaphore(max_total) if max_total else None

    async def execute(
        self, calls: Sequence[ToolCall], registry: ToolRegistry, context: ToolExecutionContext
    ) -> list[ToolResult]:
        exec_id = short_id(context.execution_id)
        results: list[ToolResult | None] = [None] * len(calls)
        contexts: list[ToolContext | None] = [None] * len(calls)
        started: list[float] = [0.0] * len(calls)
        ran: list[bool] = [False] * len(calls)

        # 1. Prepare, in order.
        runnable: list[tuple[int, _PreparedCall]] = []
        for i, call in enumerate(calls):
            started[i] = time.perf_counter()
            _log_started(call, context)
            prepared, contexts[i] = await self._prepare(call, registry, context)
            if isinstance(prepared, ToolResult):
                results[i] = prepared
            else:
                runnable.append((i, prepared))

        # 2. Run: groups of consecutive safe calls, each unsafe call alone between them.
        limit = asyncio.Semaphore(context.max_parallel_tool_calls)

        async def run(i: int, prepared: _PreparedCall) -> None:
            async with limit:
                ran[i] = True
                await _emit_started(calls[i], prepared)
                if self._total is None:
                    results[i] = await self._invoke(calls[i], prepared, contexts[i], context)
                else:
                    async with self._total:
                        results[i] = await self._invoke(calls[i], prepared, contexts[i], context)

        group: list[tuple[int, _PreparedCall]] = []

        async def flush() -> None:
            if not group:
                return
            if len(group) > 1:
                logger.info("exec=%s step=%d running %d tool call(s) in parallel (max %d at once): %s",
                            exec_id, context.step, len(group), context.max_parallel_tool_calls,
                            ", ".join(calls[i].name for i, _ in group))
                await emit_event(EventType.TOOLS_PARALLEL_GROUP, {
                    "count": len(group), "max_at_once": context.max_parallel_tool_calls,
                    "tools": [calls[i].name for i, _ in group], "call_ids": [calls[i].id for i, _ in group],
                })
            async with asyncio.TaskGroup() as tasks:
                for i, prepared in group:
                    tasks.create_task(run(i, prepared))
            group.clear()

        for i, prepared in runnable:
            if prepared.tool.concurrency_safe:
                group.append((i, prepared))
                continue
            await flush()
            if any(r is not None and is_awaiting(r) for r in results[:i]):
                # Something before this call is waiting for a person; it may depend on it.
                results[i] = deferred_result(calls[i].id, calls[i].name)
                continue
            logger.debug("exec=%s step=%d tool %s is not concurrency-safe; running it alone",
                         exec_id, context.step, calls[i].name)
            await run(i, prepared)
        await flush()

        # 3. Finish, in order.
        final: list[ToolResult] = []
        for i, call in enumerate(calls):
            if is_awaiting(results[i]):
                await _report_not_finished(call, results[i], context)
                final.append(results[i])
                continue
            result = await self.hooks.run_after_tool(call, results[i], contexts[i])
            elapsed = time.perf_counter() - started[i]
            _log_finished(call, result, context, elapsed)
            await _emit_completed(call, result, elapsed, ran[i])
            final.append(result)
        return final
