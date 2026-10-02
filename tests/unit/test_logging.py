import logging

from agent_harness import Agent, AgentLimits, LLMRequest, Message, Runtime
from agent_harness.providers.litellm_provider import LiteLLMProvider
from agent_harness.testing import ScriptedProvider, calculator, call, tool_call_response


def messages(caplog, level=logging.INFO):
    return [r.getMessage() for r in caplog.records if r.levelno >= level and r.name.startswith("agent_harness")]


async def run_with_tool_error(caplog, level):
    caplog.set_level(level, logger="agent_harness")
    provider = ScriptedProvider(
        [tool_call_response(call("calculator", {"expression": "6*7"}), call("calculator", {"expression": "1/0"})), "42"]
    )
    return await Runtime(provider).run(Agent(name="a", tools=[calculator]), "6*7?")


async def test_info_logs_trace_every_step(caplog):
    result = await run_with_tool_error(caplog, logging.INFO)
    exec_id = result.execution_id[:8]
    logs = messages(caplog)
    expected = [
        "run started",
        "step=1 started",
        "step=1 LLM responded",
        "step=1 executing 2 tool call(s): calculator, calculator",
        "tool calculator ok",
        "tool calculator failed",
        "tools done: 1 ok, 1 failed",
        "step=2 final answer produced",
        "run finished: status=completed steps=2",
    ]
    for fragment in expected:
        assert any(fragment in line for line in logs), f"missing log: {fragment!r}\n" + "\n".join(logs)
    assert all(f"exec={exec_id}" in line for line in logs)


async def test_tool_failure_is_a_warning(caplog):
    await run_with_tool_error(caplog, logging.INFO)
    warnings = messages(caplog, logging.WARNING)
    assert len(warnings) == 1 and "division by zero" in warnings[0]


async def test_debug_level_adds_detail_without_errors(caplog):
    await run_with_tool_error(caplog, logging.DEBUG)
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("context built" in m for m in debug)
    assert any('args={"expression": "6*7"}' in m for m in debug)


async def test_failed_and_limited_runs_log_at_warning_or_above(caplog):
    caplog.set_level(logging.INFO, logger="agent_harness")
    provider = ScriptedProvider(fallback=lambda r: tool_call_response(call("calculator", {"expression": "1"})))
    await Runtime(provider).run(Agent(name="a", tools=[calculator], limits=AgentLimits(max_steps=1)), "loop")
    assert any("max_steps=1 reached" in m for m in messages(caplog, logging.WARNING))

    caplog.clear()
    await Runtime(ScriptedProvider([RuntimeError("boom")])).run(Agent(name="a"), "hi")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and errors[0].exc_info is not None  # unexpected errors keep their traceback


async def test_litellm_provider_never_logs_api_key(caplog):
    caplog.set_level(logging.DEBUG, logger="agent_harness")

    async def completion(**kwargs):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    provider = LiteLLMProvider("openai/m", api_key="sk-SECRET-123", completion_fn=completion)
    await provider.generate(LLMRequest(messages=[Message.user("hi")]))
    assert caplog.records and not any("sk-SECRET-123" in r.getMessage() for r in caplog.records)


def test_library_is_silent_by_default():
    handlers = logging.getLogger("agent_harness").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
