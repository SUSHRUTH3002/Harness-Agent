"""LiteLLM adapter: translation, response parsing and error mapping. No network."""

import json

import pytest

from agent_harness import (
    Agent,
    ErrorCode,
    FinishReason,
    LLMError,
    LLMProvider,
    LLMRequest,
    Message,
    Runtime,
    ToolCall,
)
from agent_harness.providers.litellm_provider import (
    LiteLLMProvider,
    from_litellm_response,
    map_error,
    to_litellm_messages,
    to_litellm_tools,
)
from agent_harness.testing import calculator


def fake_response(content=None, tool_calls=None, finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    return {
        "model": "fake-model",
        "choices": [
            {"message": {"content": content, "tool_calls": tool_calls}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def fake_tool_call(id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_message_translation_round_trip_shapes():
    call = ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}, raw_arguments='{"expression": "1+1"}')
    messages = [
        Message.system("sys"),
        Message.user("hi"),
        Message.assistant("let me", tool_calls=[call]),
        Message.tool("c1", '{"result": 2}', name="calculator"),
    ]
    assert to_litellm_messages(messages) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "let me",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"result": 2}'},
    ]


def test_assistant_without_text_sends_null_content():
    [entry] = to_litellm_messages([Message.assistant(tool_calls=[ToolCall(id="c", name="t", arguments={})])])
    assert entry["content"] is None and entry["tool_calls"][0]["function"]["arguments"] == "{}"


def test_tool_translation():
    [spec] = to_litellm_tools([calculator.to_schema()])
    assert spec["type"] == "function" and spec["function"]["name"] == "calculator"
    assert "expression" in spec["function"]["parameters"]["properties"]


def test_parse_text_response():
    response = from_litellm_response(fake_response("hello"), cost=0.001)
    assert response.message.text == "hello" and response.finish_reason is FinishReason.STOP
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.cost_usd) == (10, 5, 0.001)
    assert response.model == "fake-model"


def test_parse_tool_calls_including_malformed_and_dict_arguments():
    response = from_litellm_response(
        fake_response(
            tool_calls=[
                fake_tool_call("a", "calculator", '{"expression": "2*3"}'),
                fake_tool_call("b", "calculator", '{"expression": '),
                fake_tool_call("c", "calculator", {"expression": "1"}),
            ],
            finish_reason="tool_calls",
        )
    )
    a, b, c = response.message.tool_calls
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert a.arguments == {"expression": "2*3"}
    assert b.arguments is None and b.raw_arguments == '{"expression": '  # surfaces as INVALID_ARGS later
    assert c.arguments == {"expression": "1"}


def test_tool_calls_imply_tool_calls_finish_reason_even_if_provider_says_stop():
    response = from_litellm_response(fake_response(tool_calls=[fake_tool_call("a", "t", "{}")], finish_reason="stop"))
    assert response.finish_reason is FinishReason.TOOL_CALLS


def test_length_finish_reason():
    assert from_litellm_response(fake_response("cut", finish_reason="length")).finish_reason is FinishReason.LENGTH


def test_empty_response_raises():
    with pytest.raises(LLMError) as info:
        from_litellm_response(fake_response(None))
    assert info.value.code == ErrorCode.EMPTY_RESPONSE
    with pytest.raises(LLMError):
        from_litellm_response({"choices": []})


def make_exc(class_name, bases=(Exception,), status=None):
    cls = type(class_name, bases, {})
    exc = cls("boom")
    if status is not None:
        exc.status_code = status
    return exc


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (make_exc("RateLimitError", status=429), ErrorCode.RATE_LIMIT, True),
        (make_exc("AuthenticationError", status=401), ErrorCode.AUTH, False),
        (make_exc("ContextWindowExceededError", (make_exc("BadRequestError").__class__,)), ErrorCode.CONTEXT_WINDOW_EXCEEDED, False),
        (make_exc("Timeout"), ErrorCode.TIMEOUT, True),
        (make_exc("APIConnectionError"), ErrorCode.TRANSPORT, True),
        (make_exc("InternalServerError", status=500), ErrorCode.SERVER, True),
        (make_exc("Whatever", status=503), ErrorCode.SERVER, True),
        (make_exc("Whatever", status=404), ErrorCode.INVALID_REQUEST, False),
        (make_exc("Whatever"), ErrorCode.LLM_ERROR, False),
    ],
)
def test_error_mapping(exc, code, retryable):
    error = map_error(exc)
    assert (error.code, error.retryable) == (code, retryable)


def test_real_litellm_exceptions_map_correctly():
    litellm = pytest.importorskip("litellm")
    exc = litellm.RateLimitError("slow down", llm_provider="openai", model="m")
    assert map_error(exc).code == ErrorCode.RATE_LIMIT
    exc = litellm.ContextWindowExceededError("too long", model="m", llm_provider="openai")
    assert map_error(exc).code == ErrorCode.CONTEXT_WINDOW_EXCEEDED


async def test_generate_merges_params_and_translates():
    completion = Recorder(fake_response("ok"))
    provider = LiteLLMProvider("openai/default-model", temperature=0.1, max_tokens=100, completion_fn=completion)
    assert isinstance(provider, LLMProvider)
    request = LLMRequest(
        messages=[Message.user("hi")], tools=[calculator.to_schema()], model="anthropic/override", settings={"max_tokens": 50}
    )
    response = await provider.generate(request)
    assert response.message.text == "ok"
    [kwargs] = completion.calls
    assert kwargs["model"] == "anthropic/override"
    assert (kwargs["temperature"], kwargs["max_tokens"], kwargs["drop_params"]) == (0.1, 50, True)
    assert kwargs["tools"][0]["function"]["name"] == "calculator"


async def test_generate_without_tools_omits_tools_key_and_uses_default_model():
    completion = Recorder(fake_response("ok"))
    await LiteLLMProvider("openai/m", completion_fn=completion).generate(LLMRequest(messages=[Message.user("hi")]))
    assert "tools" not in completion.calls[0] and completion.calls[0]["model"] == "openai/m"


async def test_generate_requires_a_model():
    with pytest.raises(LLMError) as info:
        await LiteLLMProvider(completion_fn=Recorder()).generate(LLMRequest(messages=[Message.user("x")]))
    assert info.value.code == ErrorCode.INVALID_REQUEST


async def test_generate_maps_provider_errors():
    provider = LiteLLMProvider("m", completion_fn=Recorder(make_exc("RateLimitError", status=429)))
    with pytest.raises(LLMError) as info:
        await provider.generate(LLMRequest(messages=[Message.user("x")]))
    assert info.value.code == ErrorCode.RATE_LIMIT and info.value.retryable


async def test_full_run_through_runtime_with_tool_call():
    completion = Recorder(
        fake_response(tool_calls=[fake_tool_call("c1", "calculator", '{"expression": "6*7"}')], finish_reason="tool_calls"),
        fake_response("42"),
    )
    agent = Agent(name="a", instructions="Use tools.", tools=[calculator], model="openai/m")
    result = await Runtime(LiteLLMProvider(completion_fn=completion)).run(agent, "6*7?")
    assert result.ok and result.output == "42"
    second_messages = completion.calls[1]["messages"]
    assert second_messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"expression": "6*7", "result": 42})}
    assert result.usage.input_tokens == 20


async def test_real_litellm_mock_response_end_to_end():
    pytest.importorskip("litellm")
    provider = LiteLLMProvider("openai/gpt-mock", mock_response="mocked answer", api_key="not-used")
    result = await Runtime(provider).run(Agent(name="a"), "hi")
    assert result.ok and result.output == "mocked answer"
    assert result.usage.total_tokens > 0
