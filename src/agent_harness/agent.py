"""Declarative agent definition. Holds no runtime state; one Agent can serve many runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_harness.tools import Tool


class AgentLimits(BaseModel):
    # Maximum LLM calls per run. Truncation continuations count toward it.
    max_steps: int = Field(default=15, ge=1)
    # How many times to ask the model to continue after its output was cut off
    # at the output-token limit, before failing with OUTPUT_TRUNCATED.
    max_truncation_continuations: int = Field(default=3, ge=0)


class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    id: str = ""
    instructions: str = ""
    tools: list[Tool] = Field(default_factory=list)
    model: str | None = None
    model_settings: dict[str, Any] = Field(default_factory=dict)
    limits: AgentLimits = Field(default_factory=AgentLimits)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_id(self) -> Agent:
        if not self.id:
            self.id = self.name
        return self
