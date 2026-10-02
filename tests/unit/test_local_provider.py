"""LocalProvider: request assembly, response parsing, error mapping. No network."""

import pytest

from agent_harness import ErrorCode, LLMError, LLMProvider, LLMRequest, Message
from agent_harness.providers.local_provider import LocalProvider
from agent_harness.testing import calculator


def fake_response(content=None, tool_calls=None, finish_reason="stop"):
    return {
        "model": "local-model",
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
    }


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


async def test_generate_assembles_request_and_parses_response():
    completion = Recorder(fake_response("ok"))
    provider = LocalProvider("llama-3.1-8b", completion_fn=completion)
    assert isinstance(provider, LLMProvider)
    request = LLMRequest(messages=[Message.user("hi")], tools=[calculator.to_schema()], settings={"temperature": 0.2})
    response = await provider.generate(request)
    assert response.message.text == "ok"
    [kwargs] = completion.calls
    assert kwargs["model"] == "llama-3.1-8b" and kwargs["temperature"] == 0.2
    assert kwargs["tools"][0]["function"]["name"] == "calculator"
    assert kwargs["response_model"] is None  # required positionally by instructor; None = passthrough
    # No litellm-only kwargs (e.g. drop_params) leak into a plain OpenAI-compatible call.
    assert "drop_params" not in kwargs


async def test_generate_without_tools_omits_tools_key():
    completion = Recorder(fake_response("ok"))
    await LocalProvider("m", completion_fn=completion).generate(LLMRequest(messages=[Message.user("hi")]))
    assert "tools" not in completion.calls[0]


async def test_model_on_request_overrides_constructor_default():
    completion = Recorder(fake_response("ok"))
    await LocalProvider("default-model", completion_fn=completion).generate(
        LLMRequest(messages=[Message.user("hi")], model="override-model")
    )
    assert completion.calls[0]["model"] == "override-model"


async def test_generate_requires_a_model():
    with pytest.raises(LLMError) as info:
        await LocalProvider(completion_fn=Recorder()).generate(LLMRequest(messages=[Message.user("x")]))
    assert info.value.code == ErrorCode.INVALID_REQUEST


def test_base_url_required_without_completion_fn():
    with pytest.raises(LLMError) as info:
        LocalProvider("m")
    assert info.value.code == ErrorCode.INVALID_REQUEST


def test_missing_local_extra_gives_a_clear_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name in ("instructor", "openai"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"agent-harness\[local\]"):
        LocalProvider("m", base_url="http://localhost:11434/v1")


def test_map_error_is_reused_for_openai_sdk_exceptions():
    openai = pytest.importorskip("openai")
    from agent_harness.providers.litellm_provider import map_error

    exc = openai.RateLimitError("slow down", response=_fake_httpx_response(429), body=None)
    assert map_error(exc).code == ErrorCode.RATE_LIMIT and map_error(exc).retryable


def _fake_httpx_response(status_code):
    httpx = pytest.importorskip("httpx")
    return httpx.Response(status_code, request=httpx.Request("POST", "http://x"))


async def test_generate_maps_provider_errors():
    provider = LocalProvider("m", completion_fn=Recorder(RuntimeError("connection refused")))
    with pytest.raises(LLMError) as info:
        await provider.generate(LLMRequest(messages=[Message.user("x")]))
    assert info.value.code == ErrorCode.LLM_ERROR


async def test_instructor_json_mode_is_a_passthrough_without_response_model():
    """Locks in the behavior LocalProvider relies on: with `response_model=None` (required
    positionally by `AsyncInstructor.create()` -- there is no default), instructor's Mode.JSON
    handler returns the call's other kwargs unchanged, and the underlying client's `create` is
    called with exactly those kwargs -- see this module's docstring. If this test ever fails
    after an `instructor` upgrade, LocalProvider's design assumption needs re-checking first.
    """
    instructor = pytest.importorskip("instructor")
    openai_mod = pytest.importorskip("openai")

    sentinel = object()
    received = {}

    async def fake_create(**kwargs):
        received.update(kwargs)
        return sentinel

    raw_client = openai_mod.AsyncOpenAI(api_key="x", base_url="http://localhost:1/v1")
    raw_client.chat.completions.create = fake_create
    patched = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    sent = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
    result = await patched.chat.completions.create(response_model=None, **sent)

    assert result is sentinel
    assert {k: received[k] for k in sent} == sent
