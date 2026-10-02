import pytest
from pydantic import BaseModel, Field

from agent_harness import ErrorCode, FunctionTool, Tool, ToolContext, ToolResult, ValidationError, tool
from agent_harness.tools import normalize_output


def ctx(**kw):
    return ToolContext(execution_id="e", agent_id="a", call_id="c", tool_name="t", step=1, **kw)


@tool
def add(a: int, b: int = 2) -> int:
    """Add two integers."""
    return a + b


def test_schema_from_type_hints():
    schema = add.to_schema()
    assert schema.name == "add"
    assert schema.description == "Add two integers."
    assert schema.input_schema["type"] == "object"
    assert set(schema.input_schema["properties"]) == {"a", "b"}
    assert schema.input_schema["required"] == ["a"]
    assert "title" not in schema.input_schema


async def test_validate_and_execute_sync_function():
    args = add.validate_arguments({"a": "3"})  # coerced by pydantic
    assert args == {"a": 3, "b": 2}
    assert await add.execute(args, ctx()) == 5


def test_validation_errors_are_harness_errors():
    with pytest.raises(ValidationError) as info:
        add.validate_arguments({"b": 1})
    assert info.value.code == ErrorCode.INVALID_ARGS
    assert "a" in info.value.message


def test_unknown_arguments_are_rejected():
    with pytest.raises(ValidationError):
        add.validate_arguments({"a": 1, "c": 3})


async def test_async_function_and_context_injection():
    @tool(name="echo_step", description="Echo the step.")
    async def echo(text: str, context: ToolContext) -> str:
        return f"{text}@{context.step}"

    assert "context" not in echo.input_schema["properties"]
    assert await echo.execute(echo.validate_arguments({"text": "hi"}), ctx()) == "hi@1"


class SearchArgs(BaseModel):
    query: str = Field(description="What to search")
    limit: int = 5


async def test_explicit_args_model():
    @tool(args_model=SearchArgs)
    def search(args: SearchArgs) -> dict:
        return {"q": args.query, "n": args.limit}

    assert search.input_schema["properties"]["query"]["description"] == "What to search"
    assert await search.execute(search.validate_arguments({"query": "x"}), ctx()) == {"q": "x", "n": 5}


def test_args_model_requires_single_parameter():
    with pytest.raises(TypeError):
        FunctionTool(lambda a, b: None, name="bad", args_model=SearchArgs)


def test_var_args_not_supported():
    def f(*args):
        return None

    with pytest.raises(TypeError):
        FunctionTool(f)


def test_metadata_defaults_and_overrides():
    t = FunctionTool(lambda: None, name="noop", concurrency_safe=True, timeout=2.0)
    assert (t.concurrency_safe, t.timeout) == (True, 2.0)
    assert add.concurrency_safe is False and add.timeout is None


async def test_custom_tool_subclass():
    class Upper(Tool):
        name = "upper"
        description = "Uppercase text"
        input_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

        async def execute(self, arguments, ctx):
            return arguments["text"].upper()

    t = Upper()
    assert t.to_schema().input_schema["required"] == ["text"]
    assert await t.execute({"text": "a"}, ctx()) == "A"


class Point(BaseModel):
    x: int
    y: int


@pytest.mark.parametrize(
    ("value", "content", "data"),
    [
        ("plain", "plain", None),
        (None, "", None),
        ({"a": 1}, '{"a": 1}', {"a": 1}),
        ([1, 2], "[1, 2]", [1, 2]),
        (Point(x=1, y=2), '{"x": 1, "y": 2}', {"x": 1, "y": 2}),
        (3.5, "3.5", 3.5),
    ],
)
def test_normalize_output(value, content, data):
    result = normalize_output("c1", "t", value)
    assert (result.call_id, result.name, result.content, result.data, result.is_error) == ("c1", "t", content, data, False)


def test_normalize_output_passes_tool_result_through_with_ids():
    result = normalize_output("c9", "t", ToolResult(call_id="x", name="y", content="c", is_error=True))
    assert (result.call_id, result.name, result.is_error) == ("c9", "t", True)
