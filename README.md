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

## Web UI (sessions and chats in Postgres)

A browser UI over the same agent `main.py` runs. Every chat belongs to a **user** or an **incident**, and its full history (messages, tool calls, offloaded results, compaction summaries and archives) is saved in Postgres, so you can close the tab and continue the chat later. See [docs/web-ui.md](docs/web-ui.md) for how it works.

Requirements: Docker (for Postgres), Node 18+ (for the UI), and the `.env` you already use for `main.py`.

### 1. Start the database

```bash
docker compose up -d                 # Postgres 16 on localhost:5433 (user/password/db: harness)
docker compose ps                    # STATUS should say "healthy"
```

| Task | Command |
|---|---|
| Stop it (data is kept) | `docker compose down` |
| Stop it and **delete all chats** | `docker compose down -v` |
| Open a SQL shell | `docker exec -it harness-postgres psql -U harness -d harness` |
| See the logs | `docker compose logs -f postgres` |

The server creates its tables on startup ([server/schema.sql](server/schema.sql)); there's no separate migration step. To use another Postgres instead, set `HARNESS_DATABASE_URL` in `.env` (default: `postgresql://harness:harness@localhost:5433/harness`).

### 2. Install

```bash
.venv/bin/pip install -e '.[litellm,server]'      # FastAPI, uvicorn, asyncpg
cd ui && npm install && cd ..
```

### 3. Run (development: two terminals)

```bash
# terminal 1: API on http://localhost:8000
.venv/bin/uvicorn server.app:app --reload --port 8000

# terminal 2: UI on http://localhost:5173 (forwards /api to :8000)
cd ui && npm run dev
```

Open http://localhost:5173, pick **Users** or **Incidents**, add one with **+**, then **+ New chat**.

### Run as one process (no Vite)

```bash
cd ui && npm run build && cd ..                     # writes ui/dist
.venv/bin/uvicorn server.app:app --port 8000        # serves the UI and the API at http://localhost:8000
```

### Troubleshooting

- **`could not connect to Postgres`** on server start: the database isn't running. Run `docker compose up -d` and wait for `healthy`.
- **Port 5433 already in use**: change the left side of `"5433:5432"` in `docker-compose.yml` and the port in `HARNESS_DATABASE_URL`.
- **`config error: ...`**: same `.env` checks as `main.py` (model and API key).
- **The UI says it can't reach the server**: start the API (terminal 1) before using the UI.

## Documentation

- [Repository analysis](docs/repository-analysis.md): DeepSeek Harness and TrueForge, traced from their code
- [Framework reference map](docs/framework-reference-map.md): which capability comes from where, and why
- [Web UI: sessions, chats, Postgres, live events](docs/web-ui.md)
- [Architecture](docs/architecture.md), [Design](docs/design.md), [Implementation plan](docs/implementation-plan.md), [Roadmap](docs/roadmap.md)
