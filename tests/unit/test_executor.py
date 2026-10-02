import pytest

from agent_harness import (
    ErrorCode,
    SequentialToolExecutor,
    ToolCall,
    ToolContext,
    ToolError,
    ToolExecutionContext,
    ToolRegistry,
    tool,
)
from agent_harness.testing import call

CONTEXT = ToolExecutionContext(execution_id="e1", agent_id="a1", step=3)


@tool
def double(n: int) -> int:
    """Double a number."""
    return n * 2


@tool
def explode() -> str:
    raise ValueError("kaboom")


@tool
def refuse() -> str:
    raise ToolError("not allowed here", code="CUSTOM")


@tool
def unserializable() -> object:
    class Weird:
        def __str__(self):
            raise RuntimeError("no str")

    return Weird()


@tool
def whoami(context: ToolContext) -> dict:
    return {"execution": context.execution_id, "agent": context.agent_id, "call": context.call_id, "step": context.step}


@pytest.fixture
def registry():
    return ToolRegistry([double, explode, refuse, unserializable, whoami])


async def run(registry, *calls):
    return await SequentialToolExecutor().execute(list(calls), registry, CONTEXT)


async def test_results_preserve_call_order(registry):
    results = await run(registry, call("double", {"n": 1}, id="a"), call("double", {"n": 5}, id="b"))
    assert [(r.call_id, r.content) for r in results] == [("a", "2"), ("b", "10")]
    assert results[0].data == 2


async def test_unknown_tool(registry):
    [result] = await run(registry, call("nope", id="x"))
    assert result.is_error and result.error_code == ErrorCode.UNKNOWN_TOOL
    assert "double" in result.content  # lists available tools so the model can recover


async def test_malformed_json_arguments(registry):
    [result] = await run(registry, ToolCall.parse("x", "double", '{"n": '))
    assert result.is_error and result.error_code == ErrorCode.INVALID_ARGS
    assert "not valid JSON" in result.content


async def test_non_object_arguments(registry):
    [result] = await run(registry, ToolCall.parse("x", "double", "[1]"))
    assert result.error_code == ErrorCode.INVALID_ARGS


async def test_schema_invalid_arguments(registry):
    [result] = await run(registry, call("double", {"n": "not a number"}))
    assert result.is_error and result.error_code == ErrorCode.INVALID_ARGS


async def test_missing_arguments_default_to_empty_object(registry):
    [result] = await run(registry, ToolCall(id="x", name="double"))
    assert result.error_code == ErrorCode.INVALID_ARGS  # n is required


async def test_tool_exception_becomes_error_result(registry):
    [result] = await run(registry, call("explode"))
    assert result.is_error and result.error_code == ErrorCode.TOOL_ERROR
    assert "ValueError: kaboom" in result.content


async def test_harness_error_code_is_preserved(registry):
    [result] = await run(registry, call("refuse"))
    assert (result.is_error, result.error_code, result.content) == (True, "CUSTOM", "not allowed here")


async def test_unserializable_output_becomes_error(registry):
    [result] = await run(registry, call("unserializable"))
    assert result.is_error and result.error_code == ErrorCode.TOOL_ERROR


async def test_tool_context_is_populated(registry):
    [result] = await run(registry, call("whoami", id="c7"))
    assert result.data == {"execution": "e1", "agent": "a1", "call": "c7", "step": 3}


async def test_one_result_per_call_even_when_mixed(registry):
    calls = [call("double", {"n": 2}), call("nope"), call("explode"), call("double", {"n": 3})]
    results = await run(registry, *calls)
    assert [r.call_id for r in results] == [c.id for c in calls]
    assert [r.is_error for r in results] == [False, True, True, False]
