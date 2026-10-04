# agent-harness

A lightweight, modular, application-agnostic agent harness for Python.

The harness provides the execution environment around an LLM agent:
- a runtime and lifecycle;
- an agent loop;
- messages;
- tools and a tool registry;
- a provider-neutral LLM interface;
- serializable state.

Applications (coding, research, observability, and so on) are built on top of it. The harness knows nothing about them.

**Status:** Phase 1 (core harness) is complete. See [docs/roadmap.md](docs/roadmap.md).

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python examples/basic_demo.py
.venv/bin/python -m pytest
```

```python
from agent_harness import Agent, Runtime, tool
from agent_harness.testing import ScriptedProvider, call, tool_call_response

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = Agent(name="assistant", instructions="Use tools when useful.", tools=[add])
llm = ScriptedProvider([tool_call_response(call("add", {"a": 2, "b": 3})), "2 + 3 = 5"])

result = Runtime(llm).run_sync(agent, "What is 2 + 3?")
print(result.status, result.output)   # completed 2 + 3 = 5
```

A real model plugs in by implementing `LLMProvider.generate(LLMRequest) -> LLMResponse`.

## Running with a real model (LiteLLM)

The optional LiteLLM adapter covers OpenAI, Anthropic, Gemini, DeepSeek, Azure, Bedrock, Ollama and more. You switch providers by changing the model string.

```bash
.venv/bin/pip install -e '.[dev,litellm]'
cp .env.example .env        # then set HARNESS_MODEL and the key for that provider

.venv/bin/python main.py "What's the weather in Paris in Fahrenheit?"   # one question
.venv/bin/python main.py                                                # interactive chat
.venv/bin/python main.py -q "What is 17 * 23?"                          # answer only
```

`main.py` reads `.env`. Variables already set in your shell take precedence, and blank entries are ignored. Before sending anything, it checks that the selected model's key is present. See [.env.example](.env.example) for all settings, including reliability (`HARNESS_MAX_RETRIES`, `HARNESS_MAX_EXECUTION_TIME`, `HARNESS_REPEAT_CALL_THRESHOLD`/`_ACTION`) and context budgeting (`HARNESS_MAX_CONTEXT_TOKENS`, `HARNESS_RESERVED_OUTPUT_TOKENS`).

| Provider | Model string | Env var |
|---|---|---|
| OpenAI | `openai/<model>` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/<model>` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/<model>` | `GEMINI_API_KEY` |
| DeepSeek | `deepseek/<model>` | `DEEPSEEK_API_KEY` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Ollama (local) | `ollama/<model>` | none; set `HARNESS_API_BASE=http://localhost:11434` |

In code:

```python
from agent_harness.providers.litellm_provider import LiteLLMProvider

runtime = Runtime(LiteLLMProvider("openai/<model>", temperature=0))
# Agent.model overrides the provider default per agent, so one runtime can serve agents on different providers.
```

## Documentation

- [Repository analysis](docs/repository-analysis.md): DeepSeek Harness and TrueForge, traced from their code
- [Framework reference map](docs/framework-reference-map.md): which capability comes from where, and why
- [Architecture](docs/architecture.md), [Design](docs/design.md), [Implementation plan](docs/implementation-plan.md), [Roadmap](docs/roadmap.md)
