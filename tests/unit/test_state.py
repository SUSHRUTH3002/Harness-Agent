from agent_harness import AgentState, Message, RunStatus, ToolResult, Usage
from agent_harness.testing import call


def make_state(*messages):
    state = AgentState(agent_id="a")
    for m in messages:
        state.append(m)
    return state


def test_no_pending_calls_without_assistant():
    assert make_state(Message.user("hi")).pending_tool_calls() == []


def test_pending_calls_are_unanswered_calls_of_last_assistant_message():
    c1, c2, c3 = call("a", id="c1"), call("b", id="c2"), call("c", id="c3")
    state = make_state(Message.user("go"), Message.assistant(tool_calls=[c1, c2, c3]), Message.tool("c2", "done"))
    assert [c.id for c in state.pending_tool_calls()] == ["c1", "c3"]


def test_all_answered_means_nothing_pending():
    c1 = call("a", id="c1")
    state = make_state(Message.assistant(tool_calls=[c1]), Message.tool("c1", "ok"))
    assert state.pending_tool_calls() == []


def test_only_last_assistant_message_counts():
    old = call("a", id="old")
    state = make_state(Message.assistant(tool_calls=[old]), Message.assistant("final"))
    assert state.pending_tool_calls() == []
    assert state.last_assistant_message().text == "final"


def test_state_serialization_round_trip():
    c1 = call("a", {"x": 1}, id="c1")
    state = make_state(Message.user("hi"), Message.assistant(tool_calls=[c1]), Message.tool("c1", "{}"))
    state.tool_results.append(ToolResult(call_id="c1", name="a", content="{}", data={}))
    state.usage = Usage(input_tokens=3, output_tokens=4)
    state.status = RunStatus.COMPLETED
    state.extensions["ext"] = {"k": [1, 2]}
    restored = AgentState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_terminal_statuses():
    assert not RunStatus.PENDING.is_terminal and not RunStatus.RUNNING.is_terminal
    assert all(s.is_terminal for s in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.MAX_STEPS))


def test_usage_addition():
    total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, output_tokens=4, cost_usd=0.5)
    assert (total.input_tokens, total.output_tokens, total.total_tokens, total.cost_usd) == (4, 6, 10, 0.5)
    assert (Usage() + Usage()).cost_usd is None
