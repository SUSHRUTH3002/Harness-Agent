"""End-to-end: Runtime → Agent → LLM → Tool → Tool Result → LLM → Final Result."""

import json

import pytest

from agent_harness import (
    Agent,
    AgentLimits,
    AgentState,
    ErrorCode,
    LLMError,
    Message,
    Role,
    RunStatus,
    Runtime,
    Tool,
    Usage,
    ValidationError,
)
from agent_harness.testing import (
    ScriptedProvider,
    calculator,
    call,
    get_weather,
    text_response,
    tool_call_response,
    web_search,
)


def agent(**kw):
    return Agent(name="test-agent", instructions="You are a test agent.", tools=[calculator, get_weather, web_search], **kw)


def tool_messages(request):
    return [m for m in request.messages if m.role is Role.TOOL]


async def test_single_shot_answer_without_tools():
    provider = ScriptedProvider([text_response("hello", usage=Usage(input_tokens=5, output_tokens=1))])
    result = await Runtime(provider).run(agent(), "hi")
    assert result.ok and result.status is RunStatus.COMPLETED
    assert (result.output, result.steps, result.usage.total_tokens) == ("hello", 1, 6)
    assert result.error is None
    assert result.state.started_at and result.state.finished_at
    # The request carried system prompt, user input and every tool schema.
    [request] = provider.requests
    assert [m.role for m in request.messages] == [Role.SYSTEM, Role.USER]
    assert {t.name for t in request.tools} == {"calculator", "get_weather", "web_search"}


async def test_tool_call_then_final_answer_uses_tool_result():
    def answer(request):
        [tool_msg] = tool_messages(request)
        return text_response(f"The answer is {json.loads(tool_msg.text)['result']}")

    provider = ScriptedProvider([tool_call_response(call("calculator", {"expression": "6*7"}, id="c1")), answer])
    result = await Runtime(provider).run(agent(), "What is 6*7?")
    assert result.ok and result.output == "The answer is 42"
    assert result.steps == 2
    assert [m.role for m in result.state.messages] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert result.state.tool_results[0].data == {"expression": "6*7", "result": 42}
    assert result.state.pending_tool_calls() == []


async def test_multiple_tool_calls_in_one_step():
    provider = ScriptedProvider(
        [
            tool_call_response(
                call("get_weather", {"city": "Tokyo"}, id="w"),
                call("web_search", {"query": "tokyo", "max_results": 2}, id="s"),
            ),
            lambda request: text_response(f"{len(tool_messages(request))} results"),
        ]
    )
    result = await Runtime(provider).run(agent(), "Tokyo?")
    assert result.output == "2 results"
    assert [r.call_id for r in result.state.tool_results] == ["w", "s"]


async def test_multi_step_chain():
    provider = ScriptedProvider(
        [
            tool_call_response(call("get_weather", {"city": "London"})),
            tool_call_response(call("calculator", {"expression": "14 * 9 / 5 + 32"})),
            "London is 57.2°F",
        ]
    )
    result = await Runtime(provider).run(agent(), "London in F?")
    assert result.ok and result.steps == 3 and len(result.state.tool_results) == 2


async def test_model_recovers_from_tool_error():
    def recover(request):
        [tool_msg] = tool_messages(request)
        assert tool_msg.is_error
        return tool_call_response(call("get_weather", {"city": "Paris"}))

    provider = ScriptedProvider(
        [tool_call_response(call("get_weather", {"city": "Atlantis"})), recover, "Paris is sunny"]
    )
    result = await Runtime(provider).run(agent(), "weather?")
    assert result.ok and result.output == "Paris is sunny"
    assert [r.is_error for r in result.state.tool_results] == [True, False]


async def test_invalid_tool_and_malformed_arguments_do_not_crash_the_run():
    from agent_harness import ToolCall

    provider = ScriptedProvider(
        [
            tool_call_response(call("does_not_exist", id="u"), ToolCall.parse("m", "calculator", '{"expr')),
            "done",
        ]
    )
    result = await Runtime(provider).run(agent(), "go")
    assert result.ok
    codes = [r.error_code for r in result.state.tool_results]
    assert codes == [ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGS]


async def test_max_steps_stops_a_looping_model():
    provider = ScriptedProvider(fallback=lambda r: tool_call_response(call("calculator", {"expression": "1+1"})))
    result = await Runtime(provider).run(agent(limits=AgentLimits(max_steps=4)), "loop forever")
    assert result.status is RunStatus.MAX_STEPS and not result.ok
    assert result.steps == 4 and len(provider.requests) == 4
    # Every tool call still got its result: the transcript is valid.
    assert result.state.pending_tool_calls() == []


def test_default_max_steps_is_15():
    assert AgentLimits().max_steps == 15


@pytest.mark.parametrize(
    ("exc", "code", "type_name"),
    [
        (LLMError("rate limited", code=ErrorCode.RATE_LIMIT, retryable=True), "RATE_LIMIT", "LLMError"),
        (RuntimeError("provider exploded"), "UNKNOWN", "RuntimeError"),
    ],
)
async def test_llm_failure_is_reported_not_raised(exc, code, type_name):
    result = await Runtime(ScriptedProvider([exc])).run(agent(), "hi")
    assert result.status is RunStatus.FAILED
    assert (result.error.code, result.error.type) == (code, type_name)
    assert result.state.error == result.error


async def test_llm_failure_after_tool_step_keeps_history():
    provider = ScriptedProvider([tool_call_response(call("calculator", {"expression": "1+1"})), LLMError("down")])
    result = await Runtime(provider).run(agent(), "hi")
    assert result.status is RunStatus.FAILED
    assert [m.role for m in result.state.messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]


async def test_resume_executes_pending_tool_calls_before_calling_llm():
    state = AgentState(agent_id="test-agent")
    state.append(Message.user("6*7?"))
    state.append(Message.assistant(tool_calls=[call("calculator", {"expression": "6*7"}, id="p1")]))
    provider = ScriptedProvider(["42"])
    result = await Runtime(provider).run(agent(), state=state)
    assert result.ok and result.output == "42"
    [request] = provider.requests
    assert [m.tool_call_id for m in tool_messages(request)] == ["p1"]


async def test_continuing_a_conversation_appends_to_history():
    runtime = Runtime(ScriptedProvider(["first", "second"]))
    first = await runtime.run(agent(), "one")
    second = await runtime.run(agent(), "two", state=first.state)
    assert second.ok and second.output == "second" and second.steps == 1
    assert second.execution_id == first.execution_id
    assert [m.text for m in second.state.messages] == ["one", "first", "two", "second"]


async def test_new_input_with_pending_tool_calls_is_rejected():
    state = AgentState(agent_id="test-agent")
    state.append(Message.assistant(tool_calls=[call("calculator", {"expression": "1"})]))
    with pytest.raises(ValidationError):
        await Runtime(ScriptedProvider()).run(agent(), "new question", state=state)


async def test_caller_errors_raise():
    runtime = Runtime(ScriptedProvider())
    with pytest.raises(ValidationError):
        await runtime.run(agent())  # no input for a new run
    with pytest.raises(ValidationError):
        await runtime.run(agent(), state=AgentState(agent_id="someone-else"))
    with pytest.raises(TypeError):
        Runtime(object())  # type: ignore[arg-type]


class Resource(Tool):
    name = "resource"

    def __init__(self, fail_setup=False):
        self.events = []
        self.fail_setup = fail_setup

    async def setup(self):
        if self.fail_setup:
            raise RuntimeError("cannot connect")
        self.events.append("setup")

    async def teardown(self):
        self.events.append("teardown")

    async def execute(self, arguments, ctx):
        self.events.append("execute")
        return "ok"


async def test_tool_lifecycle_runs_and_cleans_up_even_on_failure():
    res = Resource()
    provider = ScriptedProvider([tool_call_response(call("resource")), LLMError("down")])
    result = await Runtime(provider).run(Agent(name="r", tools=[res]), "go")
    assert result.status is RunStatus.FAILED
    assert res.events == ["setup", "execute", "teardown"]


async def test_tool_setup_failure_fails_the_run_cleanly():
    provider = ScriptedProvider(["never used"])
    result = await Runtime(provider).run(Agent(name="r", tools=[Resource(fail_setup=True)]), "go")
    assert result.status is RunStatus.FAILED and result.error.message == "cannot connect"
    assert provider.requests == []


def test_run_sync():
    result = Runtime(ScriptedProvider(["sync ok"])).run_sync(agent(), "hi")
    assert result.ok and result.output == "sync ok"
