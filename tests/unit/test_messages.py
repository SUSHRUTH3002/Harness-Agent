import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_harness import Message, Role, TextPart, ToolCall, ToolCallPart
from agent_harness.testing import call


def test_constructors_set_roles_and_text():
    assert Message.system("s").role is Role.SYSTEM
    assert Message.user("hi").text == "hi"
    tool_msg = Message.tool("c1", "out", name="t", is_error=True)
    assert (tool_msg.role, tool_msg.tool_call_id, tool_msg.is_error, tool_msg.text) == (Role.TOOL, "c1", True, "out")


def test_assistant_with_tool_calls():
    c = call("calculator", {"expression": "1+1"})
    msg = Message.assistant("thinking", tool_calls=[c])
    assert msg.text == "thinking"
    assert msg.tool_calls == [c]
    assert isinstance(msg.parts[0], TextPart) and isinstance(msg.parts[1], ToolCallPart)


def test_assistant_without_text_has_only_tool_call_parts():
    msg = Message.assistant(tool_calls=[call("x")])
    assert msg.text == ""
    assert len(msg.parts) == 1


def test_messages_are_immutable():
    msg = Message.user("hi")
    with pytest.raises(PydanticValidationError):
        msg.role = Role.ASSISTANT  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": Role.TOOL, "parts": [TextPart(text="x")]},  # missing tool_call_id
        {"role": Role.USER, "parts": [TextPart(text="x")], "tool_call_id": "c1"},
        {"role": Role.USER, "parts": [ToolCallPart(call=ToolCall(id="1", name="t"))]},
    ],
)
def test_role_invariants(kwargs):
    with pytest.raises(PydanticValidationError):
        Message(**kwargs)


def test_json_round_trip_preserves_parts():
    msg = Message.assistant("a", tool_calls=[call("t", {"k": 1})], metadata={"source": "model"})
    restored = Message.model_validate_json(msg.model_dump_json())
    assert restored == msg
    assert restored.tool_calls[0].arguments == {"k": 1}


def test_tool_call_parse():
    assert ToolCall.parse("1", "t", '{"a": 1}').arguments == {"a": 1}
    assert ToolCall.parse("1", "t", "").arguments == {}
    broken = ToolCall.parse("1", "t", '{"a": ')
    assert broken.arguments is None and broken.raw_arguments == '{"a": '
    assert ToolCall.parse("1", "t", "[1, 2]").arguments is None
