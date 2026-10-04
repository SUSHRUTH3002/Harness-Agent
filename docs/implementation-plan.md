# Implementation Plan

Status: Phase 0 proposal. Each phase ends with a **validation gate**: unit tests, integration tests, an architecture check, a dependency review, doc updates, and an extension-point check. We move to the next phase only after review.

## Global rules

- The core (`src/agent_harness/`) depends only on `pydantic`.
- A test (`tests/test_architecture.py`) asserts that the core imports nothing from extensions or providers, and nothing from third-party modules except pydantic.
- No module-level mutable state. Every registry and bus is owned by an instance.
- Every public type is exported from `agent_harness/__init__.py`.
- Tests use `ScriptedProvider`, a deterministic mock LLM that ships in `agent_harness.testing`. No network calls in tests.

---

## Phase 1: Core Agent Harness

**Deliverables:**

| File | Contents |
|---|---|
| `pyproject.toml` | hatchling or setuptools, src layout; Python ≥ 3.11; dependency `pydantic>=2`; dev extras `pytest`, `pytest-asyncio` |
| `messages.py` | `Role`, `TextPart`, `ToolCall`, `ToolCallPart`, `Message` + constructors and properties |
| `llm.py` | `LLMProvider`, `LLMRequest`, `LLMResponse`, `Usage`, `FinishReason`, `ToolSchema` |
| `tools.py` | `Tool` ABC, `ToolContext`, `ToolResult`, `ToolAnnotations`, `FunctionTool`, `@tool` (schema from type hints or a Pydantic args model) |
| `registry.py` | `ToolRegistry` |
| `executor.py` | `ToolExecutor` protocol, `SequentialToolExecutor` (argument parsing, validation, error-to-result conversion) |
| `context.py` | `ContextManager` protocol, minimal `DefaultContextManager` |
| `state.py` | `RunStatus`, `AgentState` (incl. `pending_tool_calls()`) |
| `agent.py` | `Agent`, `AgentLimits` |
| `loop.py` | `AgentLoop` |
| `runtime.py` | `Runtime`, `RunResult`, and `Execution` (Phase 1 exposes `run()` only; `start()` arrives in Phase 2 with cancellation) |
| `errors.py` | Base hierarchy + `ErrorInfo` |
| `testing/` | `ScriptedProvider` (a sequence of responses, or a callable per request), `mock_tools.py` (calculator, weather, search) |
| `examples/basic_demo.py` | Runtime → Agent → LLM → tool call → result → LLM → final answer, using `ScriptedProvider` |

### Tests

**Unit tests:**
- **Messages:** construction, discriminated parts, JSON round-trip, `tool_calls` and `text` accessors.
- **AgentState:** append; `pending_tool_calls()` covering none, some and all resolved; serialization round-trip.
- **Tool / `@tool`:** schema generation, validation with type hints and with a Pydantic model, sync and async functions, return normalization.
- **ToolRegistry:** register, duplicate rejection, invalid names, lookup, schemas, unregister.
- **SequentialToolExecutor:** order preserved; unknown tool → `UNKNOWN_TOOL`; malformed JSON and invalid arguments → `INVALID_ARGS`; exception → `TOOL_ERROR`; always exactly one result per call.
- **DefaultContextManager:** system message first, history in order, tools attached.
- **ScriptedProvider:** behaves deterministically.

**Integration tests:**
- A single-shot answer, with no tools.
- One tool call followed by a final answer.
- Multiple tool calls in one step.
- Multiple steps.
- A tool error that the model recovers from.
- The `max_steps` cap is reached and gives `MAX_STEPS`.
- An LLM exception gives `FAILED` with an `ErrorInfo` and no raw traceback in the result.
- Resuming from a state with pending tool calls executes them before the next LLM call.

**Failure tests (the Phase 1 subset):** LLM failure, tool failure, malformed arguments, invalid tool, maximum steps. Timeout, cancellation and context overflow wait for Phases 2 and 3.

### Exit criteria
- The demo runs.
- All tests pass.
- The architecture test passes.
- The loop body matches the conceptual shape in [design.md §8](design.md).

---

### Phase 1 outcome

- **Implemented** in `src/agent_harness/` as planned. `examples/basic_demo.py` runs.
- **91 tests** (unit + integration + architecture) pass, including with `-W error`.
- **Added beyond the plan** (Phase 0 review decision 3): continuation after output truncation, up to 3 times. See design.md §8a and `tests/integration/test_truncation.py`.
- **Deviations from the design sketch:**
  - `ToolExecutor.execute` takes the registry explicitly.
  - `RunStatus` has only the Phase 1 values.
  - `Execution` / `Runtime.start()` moved to Phase 2, where cancellation gives them a purpose.
  - `FinishReason` has no `ERROR` value; providers raise `LLMError` instead.
- **Behaviours to note:**
  - Sync tool functions run in a worker thread (`asyncio.to_thread`), so they cannot block the event loop.
  - Function-tool argument models forbid unknown fields, so hallucinated arguments give `INVALID_ARGS`.
  - `max_steps` counts LLM calls per `run()` call; continuing a conversation starts a fresh budget.

## Phase 2: Reliability and Execution Control

**Build:**
- `RetryPolicy`: `max_retries`, exponential backoff with jitter, retryable codes, honoring `retry_after`. Applied as LLM-call middleware.
- Timeouts: LLM, tool, and overall execution (`asyncio.timeout`).
- `CancellationToken` and `Runtime.start() → Execution.cancel()`.
- Transcript repair: synthetic error results for unfinished calls on cancel, timeout or error (DS `ToolCallRecovery` / TF `OpenToolCallCloser`).
- Repeated identical tool-call detection: a policy with `warn` (inject a reminder) or `stop`.
- Full error classification.

**Tests:**
- Retry succeeds after transient errors.
- A non-retryable error fails immediately.
- Backoff timing is correct (use an injectable clock or sleep).
- Each kind of timeout.
- Cancellation mid-LLM and mid-tool.
- The repeat detector.
- Transcripts stay valid after an abort.

**Gate question:** is middleware enough, or do we need the Phase 4 hook system earlier? The expected answer is middleware.

### Phase 2 outcome

- **Implemented**: `cancellation.py` (`CancellationToken`), `retry.py` (`RetryPolicy`, `ResilientLLMProvider`, composed by the caller -- not auto-wrapped by `Runtime`), per-agent `llm_timeout`/`tool_timeout`/`max_execution_time` on `AgentLimits`, `Runtime.start()` → `Execution` (`.cancel()`/`.wait()`, `run()` is now `await self.start(...).wait()`), transcript repair on cancel/timeout/failure (`_repair_transcript`), and the repeated-call guard (`AgentLoop._check_repeated_calls`, state kept in `state.extensions["repeat_guard"]`).
- **No tests written** (explicit instruction); verified by hand via ad hoc scripts covering: retry-then-succeed, non-retryable failing immediately, `max_steps`, repeat-guard `warn` and `stop`, cancellation mid-run, per-tool timeout, and `max_execution_time`. All behaved as designed.
- **Deviations from the design sketch:** `llm_timeout` is enforced per-agent inside the loop (`AgentLoop._generate`), not only via a provider wrapper -- a provider can't see which `Agent` is running, so a per-agent override has to live where `agent` is in scope. `CancelledError` (the design.md sketch's planned `HarnessError` subclass) was not added; real `asyncio.CancelledError` is used directly and converted to `RunStatus.CANCELLED` at the `Runtime` boundary, to avoid two "CancelledError" names meaning different things.
- **main.py wiring:** `HARNESS_MAX_RETRIES`, `HARNESS_MAX_EXECUTION_TIME`, `HARNESS_REPEAT_CALL_THRESHOLD`/`_ACTION`; `HARNESS_TIMEOUT` now also feeds `agent.limits.llm_timeout`. Ctrl-C during `ask()` triggers `Execution.cancel()` (verified against a real SIGINT in a subprocess: a clean `RunResult(status=cancelled)`, not a crash).

## Phase 3: Context Management

**Build:**
- `PromptSection` ordering.
- `TokenCounter` protocol (chars/4 by default; a tiktoken-style adapter as an optional extra).
- `ContextBudget` (`max_context_tokens`, `reserved_output_tokens`).
- Overflow detection → `ContextError(CONTEXT_OVERFLOW)`.
- A simple reduction: drop the oldest complete tool-call groups, and never split a call from its result.
- A `context.created` metadata record (token estimate per section, which follows TF's usage attribution).

**Tests:** section ordering, estimates, reduction preserves tool pairing and the system prompt, overflow is raised when reduction cannot fit.

### Phase 3 outcome

- **Implemented** in `context.py`: `TokenCounter` protocol + `CharTokenCounter` (chars/4), `PromptSection` (ordered, defaults to one "instructions" section), and budgeting in `DefaultContextManager` (same class, extended in place per the design doc, not a new class). Reduction drops the *oldest* complete tool-call group (`_segment_history` pairs an assistant tool-call message with the tool messages answering it) until `max_context_tokens - reserved_output_tokens` is met; index 0 (the first message) is never eligible. Raises `ContextError(CONTEXT_OVERFLOW)` if it still doesn't fit. `max_context_tokens`/`reserved_output_tokens` live on `AgentLimits`, not the `ContextManager` constructor, to stay consistent with how every other limit is per-agent.
- **No tests written** (explicit instruction); verified by hand: no-budget (nothing dropped), generous budget (nothing dropped), tight budget (drops oldest groups, keeps first and last message), and an impossible budget (raises `ContextError` with `groups_dropped` in `details`).
- **Deviations:** no formal `context.created` event/record (Phase 5's event bus doesn't exist yet) -- the same information is attached to `LLMRequest.metadata["context"]` and logged at DEBUG instead. A tiktoken-style counter was not built (optional, not requested); any real tokenizer can be swapped in via the `token_counter=` constructor argument.
- **main.py wiring:** `HARNESS_MAX_CONTEXT_TOKENS` (blank = Phase 1 behavior, unlimited), `HARNESS_RESERVED_OUTPUT_TOKENS`.

## Phase 4: Tool Execution Pipeline

**Build:**
- The `Hooks` registry with `before_tool` (`allow` / `deny` / `ask`), `wrap_tool`, `after_tool`, plus `before_step`, `transform_request`, `wrap_llm` and `before_stop`.
- `PipelineToolExecutor` with the stages validation → pre → execute → result processing → post → observation.
- Ordering rule: policy stages run serially in model order; the stage that runs tool bodies can later run concurrently.

**Tests:**
- Hook ordering.
- A deny skips the body but still produces a result.
- A post hook replaces content.
- A failing hook becomes an error result, not a crash.
- The loop is untouched: `git diff loop.py` should be limited to hook-point calls.

## Phase 5: Event System

**Build:** the `Event` model, `EventBus` (async fan-out, isolated failures, ordered), and emits from the runtime, loop, executor and context manager.

**Tests:**
- The event sequence for a scripted run matches a golden list.
- A failing observer does not affect the run.
- Every event carries its IDs.

## Phase 6: Observability

**Build:**
- `MetricsObserver`, covering runtime duration and status, LLM latency, tokens and cost, tool latency, success and result size, and steps and tool-call counts.
- `LoggingObserver`.
- An optional OTel adapter in extensions.

**Tests:** metrics computed from a scripted run.

## Phase 7: Large Result Handling

**Build:**
- The `ResultProcessor` strategies: inline, truncate (head and tail), summarize (callable), offload.
- The `ArtifactStore` protocol (in-memory and file implementations).
- A per-call and per-step token budget with largest-first offloading.
- A `read_artifact` tool, so offloading works without a sandbox, which fixes a TF limitation.

**Gate question:** is summarize worth shipping, or should it only be an interface?

## Phase 8: Streaming

**Build:**
- `Execution.events()`, an async iterator backed by a queue observer.
- `StreamingLLMProvider.stream()` with `LLMStreamEvent` (text delta, tool-call delta, usage, finish).
- A transient `llm.token` event.

**Tests:** event ordering during streaming; token events are never passed to persistence.

## Phase 9: Persistence

**Build:**
- The `PersistenceStore` protocol: `save_checkpoint(state)`, `append_events(events)`, `load(execution_id)`, `list()`.
- `InMemoryStore` and `JsonlFileStore`.
- A `PersistenceObserver` that checkpoints at `agent.step.completed` and at completion.
- A **store contract test suite** that every implementation must pass (TF pattern).

## Phase 10: Compaction

**Build:**
- The `Compactor` protocol, and `PruneToolResultsCompactor` (model-free).
- `SummarizingCompactor`:
  - selects a region, keeping a recent tail and tool pairing;
  - summarizes using the provider;
  - stores a `CompactionCheckpoint` in `state.extensions`, which the `ContextManager` applies as a view.
- Triggered by the budget, and on `CONTEXT_WINDOW_EXCEEDED` followed by a single retry.

## Phase 11: Subagents

**Build:** a `SubagentTool(agent, runtime)`:
- the child runs with fresh state;
- `parent_execution_id` is set on the child's events;
- depth limit, timeout, and cancellation linked to the parent;
- the child's result becomes the tool result;
- the full child state is available through events or persistence.

## Phase 12: Parallel Execution

**Build:**
- `ParallelToolExecutor`: tools with `concurrency_safe` run in a bounded semaphore pool; other tools act as barriers; results are committed in model order; each call's failure is isolated.
- A `gather_agents` helper for concurrent subagents.

## Phase 13: MCP

**Build:**
- `agent_harness_ext.mcp`: an `MCPToolSource` that connects over stdio or streamable HTTP, lists the server's tools, wraps each one as a `Tool` named `mcp__server__tool`, and has timeouts and reconnect.
- Optional deferred loading through meta-tools.
- `mcp` is an optional extra.

## Phase 14: Safety and Governance

**Build:**
- A `ToolPolicy`: allow and deny lists, and rules based on annotations such as "approve `destructive`".
- ~~`ApprovalHandler` protocol~~: **deferred** (HITL not required for now, per Phase 0 review).
- An optional `AWAITING_INPUT` suspend-and-resume path.
- Resource limits.
- The `Sandbox` protocol (interface only).

## Phase 15: Replay and Advanced Persistence

**Build:**
- Rebuild `AgentState` from events or checkpoints.
- Fork at a step boundary.
- A deterministic replay harness that uses recorded LLM responses, so debugging and regression tests use no model calls.

## Phase 16: Plugin / Extension Architecture

**Build:**
- A `Plugin` protocol: `name`, and `setup(registrar)` that registers hooks, tools, observers and prompt sections, with teardown.
- A `LoopStrategy` protocol.
- Documentation of every extension point.
- An audit showing that each earlier extension can be expressed as a plugin.
