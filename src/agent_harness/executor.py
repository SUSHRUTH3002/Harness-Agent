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
from agent_harness.messages import ToolCall, preview, short_id
from agent_harness.registry import ToolRegistry
from agent_harness.tools import ToolContext, ToolResult, normalize_output

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


class SequentialToolExecutor:
    """Runs calls one at a time, in the order the model issued them."""

    async def execute(
        self, calls: Sequence[ToolCall], registry: ToolRegistry, context: ToolExecutionContext
    ) -> list[ToolResult]:
        return [await self.execute_one(call, registry, context) for call in calls]

    async def execute_one(self, call: ToolCall, registry: ToolRegistry, context: ToolExecutionContext) -> ToolResult:
        exec_id = short_id(context.execution_id)
        started = time.perf_counter()
        result = await self._execute_one(call, registry, context)
        elapsed = time.perf_counter() - started
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
        return result

    async def _execute_one(self, call: ToolCall, registry: ToolRegistry, context: ToolExecutionContext) -> ToolResult:
        logger.info(
            "exec=%s step=%d tool %s started args=%s",
            short_id(context.execution_id), context.step, call.name,
            preview(json.dumps(call.arguments) if call.arguments is not None else str(call.raw_arguments)),
        )

        tool = registry.get(call.name)
        if tool is None:
            available = ", ".join(registry.names()[:_MAX_LISTED_TOOLS]) or "none"
            return ToolResult.error(
                call.id, call.name, f"Unknown tool '{call.name}'. Available tools: {available}", ErrorCode.UNKNOWN_TOOL
            )

        try:
            arguments = tool.validate_arguments(parse_arguments(call))
        except HarnessError as exc:
            return ToolResult.error(call.id, call.name, exc.message, exc.code)

        ctx = ToolContext(
            execution_id=context.execution_id,
            agent_id=context.agent_id,
            call_id=call.id,
            tool_name=call.name,
            step=context.step,
            cancel=context.cancel,
            metadata=dict(context.metadata),
        )
        timeout = tool.timeout if tool.timeout is not None else context.default_tool_timeout
        try:
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    value = await tool.execute(arguments, ctx)
            else:
                value = await tool.execute(arguments, ctx)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return ToolResult.error(
                call.id, call.name, f"Tool '{call.name}' timed out after {timeout:.0f}s", ErrorCode.TIMEOUT
            )
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
