"""agent_harness: a lightweight, modular, application-agnostic agent harness."""

import logging as _logging

# Library convention: silent unless the application configures logging.
# Every module logs under "agent_harness.<module>", so logging.getLogger("agent_harness") controls them all.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from agent_harness.agent import Agent, AgentLimits
from agent_harness.cancellation import CancellationToken
from agent_harness.context import (
    CharTokenCounter,
    ContextManager,
    DefaultContextManager,
    PromptSection,
    TokenCounter,
)
from agent_harness.errors import (
    ContextError,
    ErrorCode,
    ErrorInfo,
    ExecutionError,
    HarnessError,
    HarnessTimeoutError,
    LLMError,
    ToolError,
    ValidationError,
)
from agent_harness.executor import SequentialToolExecutor, ToolExecutionContext, ToolExecutor
from agent_harness.llm import FinishReason, LLMProvider, LLMRequest, LLMResponse, ToolSchema, Usage
from agent_harness.loop import AgentLoop, LoopDependencies
from agent_harness.messages import Message, Role, TextPart, ToolCall, ToolCallPart
from agent_harness.registry import ToolRegistry
from agent_harness.retry import RetryPolicy, ResilientLLMProvider
from agent_harness.runtime import Execution, RunResult, Runtime
from agent_harness.state import AgentState, RunStatus
from agent_harness.tools import FunctionTool, Tool, ToolAnnotations, ToolContext, ToolResult, tool

__all__ = [
    "Agent",
    "AgentLimits",
    "AgentLoop",
    "AgentState",
    "CancellationToken",
    "CharTokenCounter",
    "ContextError",
    "ContextManager",
    "DefaultContextManager",
    "ErrorCode",
    "ErrorInfo",
    "Execution",
    "ExecutionError",
    "FinishReason",
    "FunctionTool",
    "HarnessError",
    "HarnessTimeoutError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LoopDependencies",
    "Message",
    "PromptSection",
    "ResilientLLMProvider",
    "RetryPolicy",
    "Role",
    "RunResult",
    "RunStatus",
    "Runtime",
    "SequentialToolExecutor",
    "TextPart",
    "Tool",
    "ToolAnnotations",
    "ToolCall",
    "ToolCallPart",
    "ToolContext",
    "ToolError",
    "ToolExecutionContext",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "TokenCounter",
    "Usage",
    "ValidationError",
    "tool",
]
