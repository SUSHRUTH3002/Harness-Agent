"""Context construction. The loop never assembles LLM requests itself."""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

from agent_harness.agent import Agent
from agent_harness.llm import LLMRequest, ToolSchema
from agent_harness.messages import Message, short_id
from agent_harness.state import AgentState

logger = logging.getLogger(__name__)


@runtime_checkable
class ContextManager(Protocol):
    async def build(self, agent: Agent, state: AgentState, tools: Sequence[ToolSchema]) -> LLMRequest: ...


class DefaultContextManager:
    """System instructions, then the full history, plus tool schemas and model settings.

    Budgeting and reduction arrive in Phase 3 behind this same interface.
    """

    async def build(self, agent: Agent, state: AgentState, tools: Sequence[ToolSchema]) -> LLMRequest:
        messages: list[Message] = []
        if agent.instructions:
            messages.append(Message.system(agent.instructions, metadata={"source": "instructions"}))
        messages.extend(state.messages)
        logger.debug(
            "exec=%s step=%d context built: %d messages (~%d chars), %d tools, model=%s",
            short_id(state.execution_id), state.step, len(messages), sum(len(m.text) for m in messages),
            len(tools), agent.model or "<provider default>",
        )
        return LLMRequest(
            messages=messages,
            tools=list(tools),
            model=agent.model,
            settings=dict(agent.model_settings),
        )
