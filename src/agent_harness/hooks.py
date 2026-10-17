"""Named extension points the agent loop and tool executor call at fixed moments.

This is a generic interception mechanism, not a guardrail system by itself --
`before_tool` happens to be guardrail-shaped (allow/deny/ask), but
`after_tool`, `before_step` and `before_stop` exist for other jobs entirely
(replacing an oversized result, injecting context, forcing one more step).
Guardrails are one thing you can build with these hooks, not the only thing.

With no hooks registered, every one of these is a no-op and behavior is
identical to not having this module at all.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Union

from agent_harness.agent import Agent
from agent_harness.messages import ToolCall
from agent_harness.state import AgentState
from agent_harness.tools import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# --- before_tool: allow / deny / ask ------------------------------------------


@dataclass(frozen=True)
class ToolDecision:
    action: Literal["allow", "deny", "ask"]
    reason: str | None = None


def allow() -> ToolDecision:
    return ToolDecision("allow")


def deny(reason: str) -> ToolDecision:
    return ToolDecision("deny", reason)


def ask(reason: str | None = None) -> ToolDecision:
    """Request human approval. No approval mechanism exists yet (Phase 0 review:
    human-in-the-loop deferred), so this currently resolves to a denial -- the
    same fail-closed default DeepSeek and TrueForge both use when no approval
    handler is configured."""
    return ToolDecision("ask", reason)


BeforeToolHook = Callable[[ToolCall, ToolContext], Union[ToolDecision, None, Awaitable[Union[ToolDecision, None]]]]


# --- after_tool: optionally replace a result ----------------------------------

AfterToolHook = Callable[
    [ToolCall, ToolResult, ToolContext], Union[ToolResult, None, Awaitable[Union[ToolResult, None]]]
]


# --- before_step: skip a step, or stop the whole run --------------------------


@dataclass(frozen=True)
class StepDecision:
    action: Literal["continue", "stop"] = "continue"
    reason: str | None = None


BeforeStepHook = Callable[[Agent, AgentState], Union[StepDecision, None, Awaitable[Union[StepDecision, None]]]]


# --- before_stop: let the run end, or force one more step --------------------


@dataclass(frozen=True)
class StopDecision:
    action: Literal["stop", "continue"] = "stop"
    message: str | None = None  # required when action == "continue"


BeforeStopHook = Callable[[Agent, AgentState], Union[StopDecision, None, Awaitable[Union[StopDecision, None]]]]


async def _call(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


@dataclass
class Hooks:
    """A plain registry: each list runs in registration order.

    `before_tool` is deny-only and short-circuits -- the first hook that
    returns something other than `allow()` wins, and no later hook can
    override it back to allow (same monotonic-guard principle both reference
    harnesses use: a guard can refuse, never force-approve).
    `after_tool` lets each hook see and replace the previous hook's result in
    turn, so ordering matters if you register more than one.
    """

    before_step: list[BeforeStepHook] = field(default_factory=list)
    before_tool: list[BeforeToolHook] = field(default_factory=list)
    after_tool: list[AfterToolHook] = field(default_factory=list)
    before_stop: list[BeforeStopHook] = field(default_factory=list)

    # Usable directly (hooks.before_tool.append(fn)) or as decorators:
    #   @hooks.on_before_tool
    #   def my_check(call, ctx): ...
    def on_before_step(self, fn: BeforeStepHook) -> BeforeStepHook:
        self.before_step.append(fn)
        return fn

    def on_before_tool(self, fn: BeforeToolHook) -> BeforeToolHook:
        self.before_tool.append(fn)
        return fn

    def on_after_tool(self, fn: AfterToolHook) -> AfterToolHook:
        self.after_tool.append(fn)
        return fn

    def on_before_stop(self, fn: BeforeStopHook) -> BeforeStopHook:
        self.before_stop.append(fn)
        return fn

    async def run_before_step(self, agent: Agent, state: AgentState) -> StepDecision:
        for fn in self.before_step:
            decision = await _call(fn, agent, state)
            if decision is not None and decision.action == "stop":
                return decision
        return StepDecision("continue")

    async def run_before_tool(self, call: ToolCall, ctx: ToolContext) -> ToolDecision:
        for fn in self.before_tool:
            decision = await _call(fn, call, ctx)
            if decision is not None and decision.action != "allow":
                return decision
        return allow()

    async def run_after_tool(self, call: ToolCall, result: ToolResult, ctx: ToolContext) -> ToolResult:
        for fn in self.after_tool:
            replacement = await _call(fn, call, result, ctx)
            if replacement is not None:
                result = replacement
        return result

    async def run_before_stop(self, agent: Agent, state: AgentState) -> StopDecision:
        for fn in self.before_stop:
            decision = await _call(fn, agent, state)
            if decision is not None and decision.action == "continue":
                return decision
        return StopDecision("stop")
