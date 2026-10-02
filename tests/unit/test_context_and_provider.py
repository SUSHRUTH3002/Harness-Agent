import pytest

from agent_harness import Agent, AgentState, DefaultContextManager, LLMError, LLMProvider, Message, Role
from agent_harness.testing import ScriptedProvider, calculator, text_response


async def test_default_context_puts_system_first_then_history():
    agent = Agent(name="a", instructions="Be brief.", tools=[calculator], model="m1", model_settings={"temperature": 0})
    state = AgentState(agent_id="a")
    state.append(Message.user("q1"))
    state.append(Message.assistant("a1"))
    request = await DefaultContextManager().build(agent, state, [calculator.to_schema()])
    assert [m.role for m in request.messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert request.messages[0].text == "Be brief."
    assert [t.name for t in request.tools] == ["calculator"]
    assert (request.model, request.settings) == ("m1", {"temperature": 0})


async def test_no_system_message_without_instructions():
    state = AgentState(agent_id="a")
    state.append(Message.user("q"))
    request = await DefaultContextManager().build(Agent(name="a"), state, [])
    assert [m.role for m in request.messages] == [Role.USER]


async def test_context_does_not_mutate_state():
    state = AgentState(agent_id="a")
    state.append(Message.user("q"))
    await DefaultContextManager().build(Agent(name="a", instructions="sys"), state, [])
    assert len(state.messages) == 1


async def test_scripted_provider_items_and_recording():
    async def responder(request):
        return text_response(f"saw {len(request.messages)}")

    provider = ScriptedProvider(["one", text_response("two"), responder, RuntimeError("boom")])
    assert isinstance(provider, LLMProvider)
    state = AgentState(agent_id="a")
    state.append(Message.user("q"))
    request = await DefaultContextManager().build(Agent(name="a"), state, [])
    assert (await provider.generate(request)).message.text == "one"
    assert (await provider.generate(request)).message.text == "two"
    assert (await provider.generate(request)).message.text == "saw 1"
    with pytest.raises(RuntimeError):
        await provider.generate(request)
    with pytest.raises(LLMError) as info:
        await provider.generate(request)
    assert info.value.code == "SCRIPT_EXHAUSTED"
    assert len(provider.requests) == 5


async def test_scripted_provider_fallback():
    provider = ScriptedProvider(fallback="again")
    state = AgentState(agent_id="a")
    state.append(Message.user("q"))
    request = await DefaultContextManager().build(Agent(name="a"), state, [])
    assert [(await provider.generate(request)).message.text for _ in range(3)] == ["again"] * 3


def test_mock_calculator_is_safe():
    assert calculator.validate_arguments({"expression": "2**3"}) == {"expression": "2**3"}
