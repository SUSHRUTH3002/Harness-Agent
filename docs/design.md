# Design: Core Interfaces and Decisions

Status: Phase 0 proposal. The sketches here are *design intent*. Names may be refined during implementation, but each contract is part of what is being reviewed.

## 1. Design decisions (ADR summary)

| # | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| D1 | Separate `Agent` (declarative) from `Runtime`/`Execution` (live) | DeepSeek's live `Agent` handle | Definitions are testable, reusable and serializable. TrueForge's `AgentDefinition` shows that this is enough. The live handle lives in `Execution`. |
| D2 | `AgentState` in memory is the source of truth during a run. Events describe what happened. | DeepSeek event sourcing, where the log is the truth | Event sourcing needs the "model-visible means logged" invariant everywhere, which is too costly for a small core. We keep state serializable and events complete, so replay (Phase 15) is still possible. |
| D3 | Message history is append-only. Context reduction is a *view* built by the `ContextManager`. | TrueForge overwrites context on compaction | Nothing is lost. Compaction and truncation stay reversible and auditable. This follows DeepSeek's `surfaceOp: replace` idea, simplified. |
| D4 | "What's next" is derived from the transcript: pending tool calls are the calls on the last assistant message that have no result. | A stored state-machine variable | From TrueForge `deriveAgentThreadState`. Resume and repair become pure functions of state. |
| D5 | Tools never raise into the loop. Every call yields exactly one `ToolResult` (`is_error` flag). | Exceptions | Both repositories do this. The model can recover, and transcripts stay valid for every provider. |
| D6 | Neutral message and LLM vocabulary. Providers translate at the edge. | OpenAI shape internally (TrueForge) | Keeps the core provider-agnostic, as in DeepSeek. |
| D7 | Keep observation (events → observers) separate from interception (hooks). | A single bus that does both | DeepSeek's dispatch modes show why: an observer must not be able to change behavior, and an interceptor must declare what it can change. |
| D8 | Constructor injection plus `typing.Protocol`. No DI container. | Cordis-style container | Python protocols give replaceability. A container would dominate a small codebase. |
| D9 | The core enforces `max_steps` and `max_execution_time`. | Leaving it to plugins (DeepSeek) | DeepSeek has no loop cap in core. Termination guarantees belong in the core. |
| D10 | Retry, timeout and policy are middleware or hooks, not loop code. | Inline retry | Both show the loop stays readable when policy is external. DeepSeek's `llm-retry` is the model. |
| D11 | Pydantic v2 for messages, state, events and tool arguments. | dataclasses + jsonschema | Validation, JSON Schema generation and serialization in one dependency. |
| D12 | Async-first (`asyncio`). A sync convenience wrapper is provided. | Sync-first | LLM and tool I/O is async-bound. Parallel tools and subagents need it. |

## 2. Messages (`messages.py`)

```python
class Role(str, Enum):
    SYSTEM = "system"; USER = "user"; ASSISTANT = "assistant"; TOOL = "tool"

class TextPart(BaseModel):     type: Literal["text"] = "text"; text: str
class ToolCall(BaseModel):
    id: str; name: str
    arguments: dict[str, Any] | None = None   # parsed, when valid JSON
    raw_arguments: str | None = None          # original provider string (DS keeps raw; malformed → tool error)
class ToolCallPart(BaseModel): type: Literal["tool_call"] = "tool_call"; call: ToolCall
# Later, without a schema break: ImagePart, ReasoningPart (opaque provider replay data kept in metadata)

Part = Annotated[TextPart | ToolCallPart, Field(discriminator="type")]

class Message(BaseModel):
    id: str                       # uuid
    role: Role
    parts: list[Part]
    tool_call_id: str | None = None   # role == TOOL
    is_error: bool = False            # role == TOOL
    metadata: dict[str, Any] = {}     # provenance ("source"), provider replay state, etc.
    created_at: datetime

    # helpers: Message.system(text), .user(text), .assistant(text, tool_calls), .tool_result(call_id, text, is_error)
    # properties: .text, .tool_calls
```

- **Why parts?** They let tool calls, images and reasoning sit alongside text without changing the schema, as in DeepSeek's content blocks.
- **Why the `tool_call_id` and `is_error` fields?** They map directly onto OpenAI, Anthropic and Gemini tool-result shapes.

## 3. LLM provider (`llm.py`)

```python
class FinishReason(StrEnum): STOP="stop"; TOOL_CALLS="tool_calls"; LENGTH="length"   # errors are raised, not returned

class Usage(BaseModel):
    input_tokens: int = 0; output_tokens: int = 0
    cache_read_tokens: int | None = None; cache_write_tokens: int | None = None
    cost_usd: float | None = None     # only if the provider reports it

class ToolSchema(BaseModel): name: str; description: str; input_schema: dict[str, Any]

class LLMRequest(BaseModel):
    messages: list[Message]            # includes the system message; built by ContextManager
    tools: list[ToolSchema] = []
    model: str | None = None
    settings: dict[str, Any] = {}      # temperature, max_tokens, ... passed through to the adapter
    metadata: dict[str, Any] = {}

class LLMResponse(BaseModel):
    message: Message                   # role == ASSISTANT
    finish_reason: FinishReason
    usage: Usage = Usage()
    model: str | None = None
    raw: Any = Field(default=None, exclude=True)

@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
    # Phase 8 (optional capability, detected with isinstance on StreamingLLMProvider):
    # def stream(self, request) -> AsyncIterator[LLMStreamEvent]
```

- **Errors.** Adapters raise `LLMError(code=..., retryable=...)` using neutral codes: `AUTH`, `RATE_LIMIT`, `CONTEXT_WINDOW_EXCEEDED`, `INVALID_REQUEST`, `SERVER`, `TIMEOUT`, `TRANSPORT`, `EMPTY_RESPONSE`. This is DeepSeek's taxonomy.
- **Why raise instead of returning errors in-band?** DeepSeek puts errors in the stream, which suits streaming. For `generate()`, raising is more idiomatic in Python. Streaming (Phase 8) will end with a terminal error event.

## 4. Tools (`tools.py`, `registry.py`, `executor.py`)

```python
class ToolContext(BaseModel):                  # what a tool may see; never the whole runtime
    execution_id: str; agent_id: str; call_id: str; step: int
    tool_name: str
    # Phase 2 adds: cancel: CancellationToken
    metadata: dict[str, Any] = {}

class ToolResult(BaseModel):
    call_id: str; name: str
    content: str                               # model-facing text (parts later)
    is_error: bool = False
    data: Any = None                           # structured value, not sent to the model
    error_code: str | None = None              # UNKNOWN_TOOL, INVALID_ARGS, TOOL_ERROR, TIMEOUT, DENIED...
    metadata: dict[str, Any] = {}

class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]               # JSON Schema
    # optional metadata used by later phases (declared now, so the interface stays stable)
    concurrency_safe: bool = False             # Phase 12 (DS isConcurrencySafe, fail-closed default)
    timeout: float | None = None               # Phase 2
    annotations: ToolAnnotations = ToolAnnotations()  # read_only / destructive (Phase 14; TF @write/@destructive)

    def validate(self, arguments: dict) -> dict: return arguments   # override or use FunctionTool
    @abstractmethod
    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> Any: ...
    def to_schema(self) -> ToolSchema: ...

class FunctionTool(Tool):   # wraps an async/sync function; args model from type hints or a pydantic model
    ...
def tool(fn=None, *, name=None, description=None, ...) -> FunctionTool: ...   # decorator
```

- **Return normalization.** `str` is passed through. A `BaseModel`, `dict` or `list` is JSON-encoded into `content` and kept in `data`. A `ToolResult` is used as returned.
- **`ToolRegistry`:**
  - `register(tool)`: rejects duplicates and invalid names (`^[a-zA-Z0-9_-]{1,64}$`, a constraint TF learned from providers).
  - `unregister(name)`, `get(name)`, `list()`, `schemas()`, `__contains__`.
  - Optional `setup()` / `teardown()` lifecycle for tools that hold resources. The runtime calls these.
- **`ToolExecutor` protocol:** `async execute(calls, registry: ToolRegistry, context: ToolExecutionContext) -> list[ToolResult]`. The registry is passed per call because it is built per run from `agent.tools`, so one executor can serve many runs.
  - Guarantee: one result per call, in call order.
  - Unknown tool → `UNKNOWN_TOOL`.
  - Unparseable or invalid arguments → `INVALID_ARGS`.
  - Exception → `TOOL_ERROR`.
  - The Phase 1 implementation is `SequentialToolExecutor`.

## 5. State (`state.py`)

```python
class RunStatus(str, Enum):
    PENDING="pending"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"
    MAX_STEPS="max_steps"
    # Phase 2 adds CANCELLED, TIMED_OUT. AWAITING_INPUT is deferred (no human-in-the-loop for now).

class AgentState(BaseModel):
    execution_id: str; agent_id: str
    status: RunStatus = RunStatus.PENDING
    messages: list[Message] = []          # append-only history (D3)
    step: int = 0
    usage: Usage = Usage()                # cumulative
    tool_results: list[ToolResult] = []   # full records incl. data/metadata
    result: str | None = None             # final answer text
    error: ErrorInfo | None = None        # structured, never a raw exception
    metadata: dict[str, Any] = {}
    extensions: dict[str, Any] = {}       # namespaced KV for extensions (TF capability_state)
    started_at / finished_at

    def pending_tool_calls(self) -> list[ToolCall]: ...   # D4
    def append(self, message: Message) -> None: ...
```

`AgentState` is mutated only by the loop and the runtime, never by tools. Extensions write under their own `extensions[<name>]` key.

## 6. Agent (`agent.py`)

```python
class AgentLimits(BaseModel):
    max_steps: int = 15                         # decided in Phase 0 review; truncation continuations count
    max_truncation_continuations: int = 3       # decided in Phase 0 review (see §8a)
    max_execution_time: float | None = None     # seconds (Phase 2)
    llm_timeout: float | None = None; tool_timeout: float | None = None

class Agent(BaseModel):
    id: str; name: str
    instructions: str = ""
    tools: list[Tool] = []                      # arbitrary_types_allowed
    model: str | None = None
    model_settings: dict[str, Any] = {}
    limits: AgentLimits = AgentLimits()
    metadata: dict[str, Any] = {}
```

## 7. Context manager (`context.py`)

```python
class ContextManager(Protocol):
    async def build(self, agent: Agent, state: AgentState, tools: list[ToolSchema]) -> LLMRequest: ...
```

- **Phase 1 (`DefaultContextManager`):** `[system(instructions)] + state.messages` plus the tool schemas.
- **Phase 3 adds:**
  - ordered prompt sections (`PromptSection(name, order, render)`);
  - a `TokenCounter` protocol (default chars/4, which both repositories use);
  - a budget with overflow detection and a simple reduction strategy (drop the oldest *whole* tool-call/result groups while keeping the system prompt and the first user message).
- **Phase 10:** when the budget is exceeded, it calls a `Compactor`.

The loop never builds messages itself.

## 8. Loop (`loop.py`)

The intended shape, which should remain recognizable in every phase:

```python
async def run(self, agent, state, deps) -> AgentState:
    while True:
        if state.step >= agent.limits.max_steps:
            return finish(state, RunStatus.MAX_STEPS)
        state.step += 1
        request = await deps.context.build(agent, state, deps.registry.schemas())
        response = await deps.llm.generate(request)
        state.append(response.message); state.usage += response.usage
        calls = response.message.tool_calls
        if not calls:
            return finish(state, RunStatus.COMPLETED, result=response.message.text)
        results = await deps.executor.execute(calls, exec_ctx(state))
        for r in results:
            state.append(Message.tool_result(r)); state.tool_results.append(r)
```

- **Phase 2** wraps the LLM call through `wrap_llm` (retry/timeout) and checks cancellation between phases.
- **Phase 4** adds hook-point calls.
- **Phase 5** adds `emit(...)` calls.

The control flow above does not change.

## 8a. Output truncation (`finish_reason = length`)

This was decided in the Phase 0 review. When a response is cut off at the output-token limit:

1. The partial text is kept in history as an assistant message with `metadata.truncated = true`, and is added to `state.partial_output`. Any tool call in the cut-off response is **dropped**, because its arguments are incomplete.
2. The loop appends a user message with `metadata.source = "harness"` and `metadata.reason = "output_truncated"`:
   - "continue exactly where you left off" for text;
   - "re-issue the tool call, keeping it shorter" for a dropped tool call.
3. Steps 1 and 2 repeat up to `max_truncation_continuations` (default 3). Each continuation is one step and counts toward `max_steps`.
4. On success, `result` is the stitched text: the partial parts plus the final part.
5. If the output is still truncated after the last continuation, the run ends `FAILED` with `ErrorInfo(code=OUTPUT_TRUNCATED)`, and `result` holds the partial text.

This is the only policy that lives inside the loop. It is small and was requested explicitly. Phase 4 may move it behind a hook.

## 9. Runtime (`runtime.py`)

```python
class Runtime:
    def __init__(self, llm: LLMProvider, *, context_manager: ContextManager | None = None,
                 tool_executor: ToolExecutor | None = None, loop: AgentLoop | None = None): ...
                 # later phases add: observers=(), hooks=None, store=None
    async def run(self, agent: Agent, input: str | Message | list[Message], *,
                  state: AgentState | None = None) -> RunResult: ...
    def run_sync(self, agent, input) -> RunResult: ...      # asyncio.run convenience
    # Phase 2: start(agent, input) -> Execution with .cancel(), .wait(); Phase 8: .events()

class RunResult(BaseModel):
    execution_id: str; status: RunStatus; output: str | None
    state: AgentState; error: ErrorInfo | None; usage: Usage; steps: int; duration: float
```

- **Lifecycle:** build the registry from `agent.tools`, run `tool.setup()`, run the loop, then run `teardown()` in `finally` for every tool that was set up.
- **`run()` never raises for agent-level failures.** It returns `RunResult(status=FAILED, error=...)`. It raises only for programmer errors, such as an invalid `Agent` or an invalid config.
- **Resume:** passing an existing `state` continues it. D4 makes this sound: if tool calls are pending, the loop executes them before calling the LLM.

## 10. Errors (`errors.py`)

```python
class HarnessError(Exception): code: str; retryable: bool = False; details: dict
class LLMError(HarnessError)          # AUTH, RATE_LIMIT, CONTEXT_WINDOW_EXCEEDED, SERVER, TIMEOUT, ...
class ToolError(HarnessError)         # raised *inside* tools optionally; converted to ToolResult
class ValidationError(HarnessError)   # INVALID_ARGS, invalid agent/config
class HarnessTimeoutError(HarnessError)   # avoid shadowing builtin TimeoutError
class ExecutionError(HarnessError)    # runtime failures (avoid shadowing builtin RuntimeError)
class ContextError(HarnessError)      # CONTEXT_OVERFLOW, OUTPUT_TRUNCATED
class CancelledError(HarnessError)
class ErrorInfo(BaseModel): code; message; type; retryable; details   # what is stored in state/results
```

- Phase 1 ships the base hierarchy. Phase 2 fills in classification and retryability.
- The names deliberately avoid shadowing the builtin `TimeoutError` and `RuntimeError`. The spec's names map to `HarnessTimeoutError` and `ExecutionError`.

## 11. Events (Phase 5, specified now for stability)

```python
class Event(BaseModel, frozen=True):
    id: str; type: str                  # "tool.completed"
    timestamp: datetime
    execution_id: str; agent_id: str
    parent_execution_id: str | None     # subagents (Phase 11)
    step: int | None
    payload: dict[str, Any]; metadata: dict[str, Any] = {}

class Observer(Protocol):
    async def on_event(self, event: Event) -> None: ...
```

- **Event types:**
  - `runtime.started|completed`
  - `agent.started|completed`
  - `agent.step.started|completed`
  - `context.created`
  - `llm.started|completed|failed|retrying`
  - `tool.started|completed|failed`
  - `error.occurred`
  - `llm.token`: transient, Phase 8, never persisted
- **Delivery:** in order, awaited per observer. A timeout or exception in an observer is logged and swallowed. Observers cannot veto (D7).
