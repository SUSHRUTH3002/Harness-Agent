"""Phase 1 demo: Runtime → Agent → LLM → tool calls → tool results → LLM → final answer.

A scripted provider stands in for a real model: its first reply requests two
tools, and its second reply is composed from the tool results it receives in
the request. Run with:  python examples/basic_demo.py
"""

from __future__ import annotations

import asyncio
import json

from agent_harness import Agent, LLMRequest, Role, Runtime
from agent_harness.testing import (
    ScriptedProvider,
    calculator,
    call,
    get_weather,
    text_response,
    tool_call_response,
    web_search,
)


def plan(request: LLMRequest):
    return tool_call_response(
        call("get_weather", {"city": "Paris"}),
        call("calculator", {"expression": "19 * 9 / 5 + 32"}),
        text="I'll check the weather and convert the temperature.",
    )


def answer(request: LLMRequest):
    results = {m.name: json.loads(m.text) for m in request.messages if m.role is Role.TOOL}
    weather, fahrenheit = results["get_weather"], results["calculator"]["result"]
    return text_response(
        f"It is {weather['condition']} in {weather['city']} at {weather['temperature_c']}°C ({fahrenheit:g}°F)."
    )


async def main() -> None:
    agent = Agent(
        name="demo-assistant",
        instructions="You are a helpful assistant. Use tools when they help.",
        tools=[calculator, get_weather, web_search],
    )
    runtime = Runtime(ScriptedProvider([plan, answer]))
    result = await runtime.run(agent, "What's the weather in Paris, in Fahrenheit?")

    for message in result.state.messages:
        label = message.role.value if message.role is not Role.TOOL else f"tool:{message.name}"
        calls = ", ".join(f"{c.name}({json.dumps(c.arguments)})" for c in message.tool_calls)
        print(f"[{label:>18}] {message.text}" + (f"  -> {calls}" if calls else ""))
    print(f"\nstatus={result.status.value} steps={result.steps} duration={result.duration * 1000:.1f}ms")
    print(f"output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
