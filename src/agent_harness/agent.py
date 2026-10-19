"""Declarative agent definition. Holds no runtime state; one Agent can serve many runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_harness.tools import Tool


class AgentLimits(BaseModel):
    # Maximum LLM calls per run. Truncation continuations count toward it.
    max_steps: int = Field(default=15, ge=1)
    # How many times to ask the model to continue after its output was cut off
    # at the output-token limit, before failing with OUTPUT_TRUNCATED.
    max_truncation_continuations: int = Field(default=3, ge=0)
    # Overall wall-clock budget for one run() call. None = unlimited.
    max_execution_time: float | None = Field(default=None, gt=0)
    # Per LLM call. None = unlimited. (Retry is separate: see retry.ResilientLLMProvider.)
    llm_timeout: float | None = Field(default=None, gt=0)
    # Default per tool call, used only when the tool itself doesn't set Tool.timeout.
    tool_timeout: float | None = Field(default=None, gt=0)
    # Consecutive steps issuing the exact same (tool name, arguments) before acting.
    # None disables the guard entirely.
    repeat_call_threshold: int | None = Field(default=3, ge=1)
    repeat_call_action: Literal["warn", "stop"] = "warn"
    # Soft cap on estimated prompt tokens. None = unlimited (Phase 1/2 behavior).
    max_context_tokens: int | None = Field(default=None, gt=0)
    # Headroom subtracted from max_context_tokens for the model's own reply.
    reserved_output_tokens: int = Field(default=1024, ge=0)

    # --- Phase 7: large result handling ----------------------------------------
    # Maximum estimated tokens for a single call's result before offloading.
    large_result_per_call_tokens: int | None = Field(default=6000, gt=0)
    # Maximum estimated tokens for a single call's result before offloading.
    large_result_per_step_tokens: int | None = Field(default=10000, gt=0)
    # Characters shown from the start and from the end of an offloaded result.
    large_result_preview_chars: int = Field(default=100, ge=0)


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
