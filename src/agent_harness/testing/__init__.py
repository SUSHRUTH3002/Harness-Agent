"""Deterministic test doubles for building and testing agents without a real LLM."""

from agent_harness.testing.mock_tools import calculator, get_weather, web_search
from agent_harness.testing.scripted import (
    ScriptedProvider,
    call,
    text_response,
    tool_call_response,
    truncated_response,
)

__all__ = [
    "ScriptedProvider",
    "calculator",
    "call",
    "get_weather",
    "text_response",
    "tool_call_response",
    "truncated_response",
    "web_search",
]
