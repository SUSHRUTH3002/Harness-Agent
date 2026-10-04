"""Context construction. The loop never assembles LLM requests itself.

Phase 3 adds ordered prompt sections and a token budget: when
`agent.limits.max_context_tokens` is set, the oldest complete tool-call
groups are dropped from the *view* sent to the model -- `state.messages`
itself is never edited (design.md D3). The system prompt and the first
message are never dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from agent_harness.agent import Agent
from agent_harness.errors import ContextError, ErrorCode
from agent_harness.llm import LLMRequest, ToolSchema
from agent_harness.messages import Message, Role, short_id
from agent_harness.state import AgentState

logger = logging.getLogger(__name__)

# Fixed per-message cost (role, structural JSON) on top of its text, matching
# the same chars/4-plus-overhead heuristic both reference harnesses use.
_MESSAGE_OVERHEAD_TOKENS = 4


@runtime_checkable
class ContextManager(Protocol):
    async def build(self, agent: Agent, state: AgentState, tools: Sequence[ToolSchema]) -> LLMRequest: ...


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharTokenCounter:
    """chars/4 heuristic. No tokenizer dependency; swap in a real one via the constructor."""

    def count(self, text: str) -> int:
        return (len(text) + 3) // 4 if text else 0


@dataclass(frozen=True)
class PromptSection:
    name: str
    order: int
    # Returns the section's text, or None/"" to omit it for this request.
    render: Callable[[Agent, AgentState], str | None]


def _instructions_section(agent: Agent, state: AgentState) -> str | None:
    return agent.instructions or None


DEFAULT_SECTIONS: tuple[PromptSection, ...] = (PromptSection("instructions", 0, _instructions_section),)


def _message_tokens(message: Message, counter: TokenCounter) -> int:
    total = counter.count(message.text) + _MESSAGE_OVERHEAD_TOKENS
    for call in message.tool_calls:
        total += counter.count(call.name) + counter.count(call.raw_arguments or "") + _MESSAGE_OVERHEAD_TOKENS
    return total


def _tool_schema_tokens(tools: Sequence[ToolSchema], counter: TokenCounter) -> int:
    return sum(
        counter.count(t.name) + counter.count(t.description) + counter.count(json.dumps(t.input_schema))
        for t in tools
    )


@dataclass
class _Segment:
    """One unit of reduction: a single kept message, or a complete tool-call group."""

    messages: list[Message]
    is_tool_group: bool


def _segment_tokens(segment: _Segment, counter: TokenCounter) -> int:
    return sum(_message_tokens(m, counter) for m in segment.messages)


def _segment_history(messages: Sequence[Message]) -> list[_Segment]:
    """Group an assistant tool-call message with the tool results that answer it.

    Everything else (plain user/assistant turns) is its own single-message,
    non-reducible segment -- the simple Phase 3 strategy only trims verbose
    tool exchanges, leaving conversation untouched (see implementation-plan.md
    Phase 3; smarter reduction is Phase 10's compaction).
    """
    segments: list[_Segment] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role is Role.ASSISTANT and message.tool_calls:
            call_ids = {c.id for c in message.tool_calls}
            group = [message]
            j = i + 1
            while j < len(messages) and messages[j].role is Role.TOOL and messages[j].tool_call_id in call_ids:
                group.append(messages[j])
                j += 1
            segments.append(_Segment(group, is_tool_group=True))
            i = j
        else:
            segments.append(_Segment([message], is_tool_group=False))
            i += 1
    return segments


class DefaultContextManager:
    """Ordered system-prompt sections, then history, plus tool schemas and model settings.

    `sections` default to just the agent's own instructions; pass your own
    `PromptSection`s to add more, ordered pieces without subclassing.
    """

    def __init__(
        self, *, sections: Sequence[PromptSection] = DEFAULT_SECTIONS, token_counter: TokenCounter | None = None
    ) -> None:
        self.sections = sorted(sections, key=lambda s: s.order)
        self.token_counter = token_counter or CharTokenCounter()

    def _render_system_prompt(self, agent: Agent, state: AgentState) -> str | None:
        parts = [text for s in self.sections if (text := s.render(agent, state))]
        return "\n\n".join(parts) if parts else None

    def _fit_budget(
        self,
        history: list[Message],
        tools: Sequence[ToolSchema],
        system: Message | None,
        budget_tokens: int,
        *,
        execution_id: str = "",
        step: int = 0,
    ) -> tuple[list[Message], int]:
        """Drop oldest complete tool-call groups until history fits `budget_tokens`. Never touches `history[0]`."""
        counter = self.token_counter
        fixed = (_message_tokens(system, counter) if system else 0) + _tool_schema_tokens(tools, counter)
        segments = _segment_history(history)
        total = fixed + sum(_segment_tokens(s, counter) for s in segments)
        over_budget_by = max(0, total - budget_tokens)
        dropped_indices: set[int] = set()
        dropped_summaries: list[str] = []

        # index 0 holds the first message (conventionally the user's initial input); never eligible.
        for i in range(1, len(segments)):
            if total <= budget_tokens:
                break
            if not segments[i].is_tool_group:
                continue
            segment_tokens = _segment_tokens(segments[i], counter)
            total -= segment_tokens
            dropped_indices.add(i)
            tool_names = ", ".join(sorted({c.name for m in segments[i].messages for c in m.tool_calls}))
            dropped_summaries.append(f"{tool_names} (~{segment_tokens} tokens)")

        if over_budget_by:
            exec_id = short_id(execution_id) if execution_id else "?"
            if dropped_summaries:
                logger.warning(
                    "exec=%s step=%d context budget exceeded by ~%d tokens (limit %d); dropped %d oldest "
                    "tool-call group(s) to fit: %s",
                    exec_id, step, over_budget_by, budget_tokens, len(dropped_summaries),
                    "; ".join(dropped_summaries),
                )
            else:
                logger.warning(
                    "exec=%s step=%d context budget exceeded by ~%d tokens (limit %d); nothing droppable "
                    "(no tool-call groups in history)",
                    exec_id, step, over_budget_by, budget_tokens,
                )

        if total > budget_tokens:
            raise ContextError(
                f"Context does not fit the {budget_tokens}-token budget even after dropping "
                f"{len(dropped_indices)} tool-call group(s) ({total} tokens remain).",
                code=ErrorCode.CONTEXT_OVERFLOW,
                details={
                    "estimated_tokens": total, "budget_tokens": budget_tokens, "groups_dropped": len(dropped_indices)
                },
            )
        reduced = [m for i, s in enumerate(segments) if i not in dropped_indices for m in s.messages]
        return reduced, len(dropped_indices)

    async def build(self, agent: Agent, state: AgentState, tools: Sequence[ToolSchema]) -> LLMRequest:
        system_text = self._render_system_prompt(agent, state)
        system = Message.system(system_text, metadata={"source": "instructions"}) if system_text else None
        history = list(state.messages)
        groups_dropped = 0

        budget = agent.limits.max_context_tokens
        if budget is not None:
            history, groups_dropped = self._fit_budget(
                history, tools, system, budget - agent.limits.reserved_output_tokens,
                execution_id=state.execution_id, step=state.step,
            )

        messages = ([system] if system else []) + history
        estimated_tokens = sum(_message_tokens(m, self.token_counter) for m in messages) + _tool_schema_tokens(
            tools, self.token_counter
        )
        logger.debug(
            "exec=%s step=%d context built: %d messages (~%d tokens est.), %d tools, model=%s%s",
            short_id(state.execution_id), state.step, len(messages), estimated_tokens, len(tools),
            agent.model or "<provider default>",
            f", dropped {groups_dropped} tool-call group(s) to fit budget" if groups_dropped else "",
        )
        return LLMRequest(
            messages=messages,
            tools=list(tools),
            model=agent.model,
            settings=dict(agent.model_settings),
            metadata={"context": {"estimated_tokens": estimated_tokens, "groups_dropped": groups_dropped}},
        )
