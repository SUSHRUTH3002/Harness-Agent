"""finish_reason=length handling: continue up to `max_truncation_continuations` times."""

from agent_harness import Agent, AgentLimits, ErrorCode, Role, RunStatus, Runtime
from agent_harness.loop import CONTINUE_PROMPT, REISSUE_TOOL_CALL_PROMPT
from agent_harness.testing import (
    ScriptedProvider,
    calculator,
    call,
    text_response,
    tool_call_response,
    truncated_response,
)


def agent(**limits):
    return Agent(name="t", tools=[calculator], limits=AgentLimits(**limits))


async def test_default_allows_three_continuations():
    assert AgentLimits().max_truncation_continuations == 3


async def test_truncated_text_is_continued_and_stitched():
    provider = ScriptedProvider([truncated_response("The quick "), truncated_response("brown fox "), "jumps."])
    result = await Runtime(provider).run(agent(), "tell me")
    assert result.ok and result.output == "The quick brown fox jumps."
    assert result.steps == 3
    # The model saw its partial output followed by a continuation request.
    last = provider.requests[-1].messages
    assert [m.role for m in last[-4:]] == [Role.ASSISTANT, Role.USER, Role.ASSISTANT, Role.USER]
    assert last[-1].text == CONTINUE_PROMPT and last[-1].metadata["reason"] == "output_truncated"
    assert result.state.messages[1].metadata["truncated"] is True
    assert result.state.consecutive_truncations == 0 and result.state.partial_output == ""


async def test_gives_up_after_three_continuations():
    provider = ScriptedProvider(fallback=lambda r: truncated_response("more "))
    result = await Runtime(provider).run(agent(), "tell me")
    assert result.status is RunStatus.FAILED
    assert result.error.code == ErrorCode.OUTPUT_TRUNCATED
    assert len(provider.requests) == 4  # original + 3 continuations
    assert result.output == "more more more more "  # partial text is kept


async def test_truncated_tool_call_is_discarded_not_executed():
    broken = call("calculator", {"expression": "1+"}, id="broken")
    provider = ScriptedProvider([truncated_response(tool_calls=[broken]), _reissue, text_response("2")])
    result = await Runtime(provider).run(agent(), "1+1?")
    assert result.ok and result.output == "2"
    executed = [r.call_id for r in result.state.tool_results]
    assert executed == ["good"] and "broken" not in executed
    assert result.state.pending_tool_calls() == []


def _reissue(request):
    assert request.messages[-1].text == REISSUE_TOOL_CALL_PROMPT
    return tool_call_response(call("calculator", {"expression": "1+1"}, id="good"))


async def test_zero_continuations_fails_immediately():
    provider = ScriptedProvider([truncated_response("cut"), "never reached"])
    result = await Runtime(provider).run(agent(max_truncation_continuations=0), "go")
    assert result.status is RunStatus.FAILED and result.error.code == ErrorCode.OUTPUT_TRUNCATED
    assert result.output == "cut" and len(provider.requests) == 1


async def test_continuations_count_toward_max_steps():
    provider = ScriptedProvider(fallback=lambda r: truncated_response("x"))
    result = await Runtime(provider).run(agent(max_steps=2, max_truncation_continuations=3), "go")
    assert result.status is RunStatus.MAX_STEPS and len(provider.requests) == 2
