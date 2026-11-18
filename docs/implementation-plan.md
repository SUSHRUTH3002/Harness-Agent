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

### Phase 4 outcome

- **Implemented** in `hooks.py`: a single `Hooks` registry (not a separate `PipelineToolExecutor` class) with four points -- `before_tool` (`allow`/`deny`/`ask`, deny-only and short-circuiting: the first non-allow decision wins and no later hook can override it back to allow), `after_tool` (each hook may replace the previous one's `ToolResult` in turn), `before_step` (stop the run before the next LLM call), `before_stop` (force one more step with an injected message instead of finishing). Registration is a plain list per hook type, in order, usable directly or as a decorator (`@hooks.on_before_tool`).
- **`transform_request` and `wrap_llm` were deliberately not built as a second hook list.** Both jobs are already fully served by composition: `ContextManager` for the former, `ResilientLLMProvider` (retry.py) for the latter. Adding a parallel hook mechanism for the same two jobs would mean two different ways to do one thing, which is the opposite of the "generic mechanism, not a second retry system" goal. `wrap_tool` was folded into the existing `Tool.timeout`/`agent.limits.tool_timeout` mechanism for the same reason, rather than becoming a third way to time a tool call.
- **No tests written** (explicit instruction); verified by hand: `before_tool` deny, `before_tool` ask (resolves to a denial with `TOOL_APPROVAL_REQUIRED`, since no approval handler exists -- same fail-closed default DeepSeek/TrueForge use), `after_tool` content replacement, `before_step` stop (confirmed the LLM is never even called), `before_stop` forced continuation. All five confirmed working with no regression to the Phase 2/3 scenarios (repeated-call guard, cancellation) or the Phase 1 demo, which all still pass unmodified with hooks left empty (the default).
- **Design note:** `after_tool` sees the *same* `ToolContext` object `before_tool` saw (fixed during implementation -- an earlier draft constructed a second, separate `ToolContext` for `after_tool`, which would have silently lost anything a `before_tool` hook stashed in `ctx.metadata` for a later `after_tool` hook to read).
- **New error codes:** `TOOL_DENIED`, `TOOL_APPROVAL_REQUIRED`, `STOPPED_BY_HOOK`.
- **Runtime wiring:** `Runtime(llm, hooks=...)`; the same `Hooks` instance is shared between the loop (`before_step`/`before_stop`) and the *default* tool executor (`before_tool`/`after_tool`) only -- a caller-supplied executor manages its own hooks wiring.
- **main.py:** one illustrative example hook (`build_hooks()`), a `before_tool` check blocking `web_search` for a couple of banned words -- clearly marked as a replaceable example, not a real content policy.

## Phase 5: Event System

**Build:** the `Event` model, `EventBus` (async fan-out, isolated failures, ordered), and emits from the runtime, loop, executor and context manager.

**Tests:**
- The event sequence for a scripted run matches a golden list.
- A failing observer does not affect the run.
- Every event carries its IDs.

**Outcome (done):**
- `events.py`: `Event` (id, type, seq, timestamp, execution_id, agent_id, parent_execution_id, step, data, transient, schema_version), `EventBus`, `Subscription` (fnmatch type filters, `unsubscribe()`), `RunEmitter`, `emit_event()`, `current_emitter()`, `JsonlEventWriter`, `PrintObserver`.
- **Open vocabulary:** `type` is a dotted string and `data` a free dict, so later phases add events without touching `events.py`. `EventType` only names the built-in ones.
- **No plumbing:** `Runtime` binds the run's `RunEmitter` to a context variable for the run's task; tasks it starts (parallel tools, worker threads) inherit it. Any code inside the run calls `emit_event(...)`; outside a run it is a no-op. Tools use `ctx.emit("progress", {...})` -> `tool.progress`.
- **Two levels of observers:** `Runtime(events=EventBus())` sees every run; `runtime.run(..., observers=[...])` sees one run (one chat's UI).
- **Delivery:** in `seq` order (a per-run lock keeps concurrent parallel-tool emits ordered), awaited per observer with `observer_timeout` (default 5s); exceptions/timeouts logged and swallowed. An observer emitting from inside `on_event` is delivered directly (no deadlock).
- **Emitted:** `run.started|completed`, `step.started|completed` (`outcome`: tool_calls / final_answer / truncated / continued_by_hook), `context.built`, `llm.started|completed|failed|retrying|truncated`, `tool.started` (when the body actually starts) / `tool.completed` (after `after_tool`, model order, `ran=false` for calls rejected in `_prepare`), `tools.parallel_group`, `tool.result_offloaded`, `compaction.started|completed|skipped|failed`, `guard.repeated_call`.
- **Reserved for later phases:** `interaction.*` (12b, now built), `llm.delta` (8, `transient=True`), `checkpoint.saved` (9, now built), `subagent.*` (11, with `parent_execution_id`).
- **Deviations from the plan:** names simplified (`run.*`/`step.*` instead of `runtime.*`/`agent.*`/`agent.step.*`); `error.occurred` folded into `run.completed.error` and `llm.failed`; `context.created` is `context.built`. Log lines are unchanged; events are additive. No tests (per the standing instruction); verified with a scratchpad script and a real Gemini run.
- **main.py:** `HARNESS_EVENTS_FILE` (JSONL) and `HARNESS_PRINT_EVENTS`.

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

### Phase 7 outcome

- **Implemented** in `results.py`: `ArtifactStore` protocol with `InMemoryArtifactStore` and `FileArtifactStore`; `LargeResultProcessor`, which offloads oversized tool results and replaces them with a head/tail preview plus a pointer; `make_read_artifact_tool(store)`, a factory for a `read_artifact` tool bound to a given store (pagination via `start`/`length`, or a `search` substring to jump to a match).
- **"Truncate" collapsed into "offload," not a separate tier.** The original four-strategy sketch (inline / truncate / summarize / offload) simplified to two outcomes in practice: inline, or offload-with-preview. A bare truncate-and-discard was never actually implemented, because offload-with-preview is strictly better at the same context cost -- it shows the same head/tail preview a truncate would, but the full content is still recoverable via `read_artifact` instead of being permanently thrown away.
- **Gate question resolved: summarize was not built.** Only inline and offload exist. Summarizing would mean an extra LLM call every time it triggers, which is a real cost/latency tradeoff the plan flagged as needing a decision -- shipping without it first was the safer default, explicitly called out as such when it was made rather than decided silently.
- **Not implemented via the Phase 4 `after_tool` hook**, despite that being the original plan going in. `after_tool` only sees one call's result at a time; the per-step *combined* threshold ("several results together are too big, even though none is individually") needs to see the whole step's batch at once to correctly offload the largest ones first. `LargeResultProcessor.process()` is instead called directly from `loop.py._execute_tools`, once per step, after the executor's per-call contract is enforced (every call has exactly one real result) and before anything is appended to state. `after_tool` hooks still run first, per call, inside the executor, and see the original un-offloaded content -- large-result handling is the last pass before a result enters history.
- **Thresholds live on `AgentLimits`** (`large_result_per_call_tokens=6000`, `large_result_per_step_tokens=10000`, `large_result_preview_chars=100` -- TrueForge's published defaults, per the Phase 0 analysis), not on `LargeResultProcessor`'s constructor, for the same reason Phase 3's context budget lives there: it needs to be read fresh per agent/per call, and a `Runtime`-level processor is constructed once and could otherwise serve agents with different needs.
- **Two-level opt-in**, matching the `hooks`/`result_processor` pattern from Phase 4: nothing runs unless a `LargeResultProcessor` is attached to `Runtime` (`result_processor=`); once attached, the `AgentLimits` defaults apply unless a given agent overrides them.
- **Bug found in use, then fixed: `read_artifact`'s own results were being offloaded.** With a low `large_result_per_call_tokens` (500), every read came back as yet another preview + pointer, so the model never saw the data and looped, creating a new artifact per attempt until `max_steps`. The repeated-call guard didn't catch it, because each attempt used different arguments. Fix: a `Tool.exempt_from_offload` flag (set on `read_artifact`), which `loop.py` passes to `LargeResultProcessor.process(exempt_tools=...)`. Exempt results are never offloaded and don't count toward the per-step total. Since `read_artifact` is now exempt, it caps its own output instead (`make_read_artifact_tool(store, max_chars=4000)`): a larger `length` is clamped and `has_more` tells the model to page.
- **Error results are never offloaded**, even if long -- the model must always see a tool failure directly, never behind an artifact pointer it would have to think to go read.
- **No tests written** (explicit instruction); verified by hand: a single oversized result offloads and is readable back via `read_artifact` (including `start`/`length` paging); two parallel results each under the per-call limit but combined over the per-step limit correctly offload only as many of the largest as needed to fit (not all of them); a processor-free `Runtime` is byte-for-byte unchanged from Phase 1-6; disabling just the per-call pass (`large_result_per_call_tokens=None`) while keeping the per-step pass works independently; a huge error result is confirmed never offloaded; `after_tool` hooks and `LargeResultProcessor` were confirmed to compose correctly together (hook sees original content, offloading is the final pass).
- **main.py wiring:** `HARNESS_LARGE_RESULT_HANDLING` (default on), `HARNESS_LARGE_RESULT_PER_CALL_TOKENS`/`_PER_STEP_TOKENS`/`_PREVIEW_CHARS`, `HARNESS_ARTIFACT_DIR` (defaults to `.artifacts/` next to `main.py`, now git-ignored). Uses `FileArtifactStore` rather than the in-memory default so offloaded results are inspectable on disk and survive past the process exiting; `read_artifact` is added to the agent's tool list automatically whenever large-result handling is enabled.

## Phase 8: Streaming

**Build:**
- `Execution.events()`, an async iterator backed by a queue observer.
- `StreamingLLMProvider.stream()` with `LLMStreamEvent` (text delta, tool-call delta, usage, finish).
- A transient `llm.token` event.

**Tests:** event ordering during streaming; token events are never passed to persistence.

### Phase 8 outcome

Decisions taken at the review gate (2026-11-18): a stream that fails partway is **discarded and retried** (with an `llm.stream_reset` event); an **idle timeout** (30s) bounds gaps between pieces on top of the overall limit; the CLI **streams by default** in a terminal (`--no-stream` to turn off); a slow `events()` reader **drops old pieces**, never durable events; `Execution.events()` is **in this phase**; a provider that reports no token counts gives **"usage unknown"**, never an estimate. The chat API (SSE) and UI are a later phase. Design: `docs/phase-8-streaming-design.md`.

- **Vocabulary** (`llm.py`): `LLMStreamEvent` with kinds `text`, `tool_call` (index, id/name once, argument fragments), `usage`, `finish` (optionally carrying a whole `LLMResponse`) and `reset`. `StreamingLLMProvider` is an `LLMProvider` with `stream()`; `generate()` stays. `LLMResponse.usage_reported` is False when the provider sent no counts.
- **One code path** (`streaming.py`): `StreamAssembler` builds the response. Tool-call arguments are joined as text and parsed only at the end; a piece naming a different call id under the same index starts a new call (some providers number every call 0). `open_stream()` replays a non-streaming provider's whole response as pieces, so callers never special-case it. `timed_stream()` adds the idle and total timeouts and always closes the source.
- **Providers**: LiteLLM and the local provider share `openai_stream_events()` (OpenAI-style chunks to pieces, `stream_options.include_usage`, errors mapped to neutral `LLMError`s; an empty or unfinished stream fails retryably). LiteLLM prices a stream from its reported tokens (`litellm.cost_per_token`); local servers report no cost.
- **Retries** (`retry.py`): `ResilientLLMProvider.stream()` uses the same policy; after pieces were passed on, a retry yields `reset` first. `idle_timeout` counts from the first piece: waiting for the first one is bounded by the provider/request timeout instead, because thinking models can be silent for several seconds (Gemini: ~6s observed) before writing.
- **Loop**: with `LoopDependencies.stream`, `_call_llm` reads pieces, emits the transient events, and assembles the response; everything after the call (tools, compaction measurement, persistence, HITL) is unchanged. Cancelling closes the model connection.
- **Runtime**: `Runtime(stream=True)` by default. `Execution.events()` is fed by an `EventStream` observer attached at `start()`, so nothing is missed by a late reader; it ends when the run's task ends. Transient events are capped (2000 by default, oldest dropped, `Execution.dropped_events`).
- **Events**: `PrintObserver` skips `llm.token` by default; `JsonlEventWriter` and the stores already skip transient events.
- **Deviations from the plan**: errors are raised from the stream as neutral `LLMError`s (the Python form of "errors arrive in the stream"); no test suite (per the standing instruction): verified with scratchpad scripts (12 streaming scenarios; Phase 5, 9 and 12b checks re-run, Phase 9 on InMemory and Postgres) and a live Gemini run.
- **Known limit**: streamed pieces are not stored, so a client that disconnects mid-answer reloads the chat and sees the finished answer, not the words it missed.

## Phase 9: Persistence

**Build:**
- The `PersistenceStore` protocol: `save_checkpoint(state)`, `append_events(events)`, `load(execution_id)`, `list()`.
- `InMemoryStore` and `JsonlFileStore`.
- A `PersistenceObserver` that checkpoints at `agent.step.completed` and at completion.
- A **store contract test suite** that every implementation must pass (TF pattern).

### Phase 9 outcome

Decisions taken at the review gate (2026-11-17): the chat id **is** `AgentState.execution_id`; a crashed run is recovered after its lease expires, closing its open tool calls as interrupted rather than re-running them; the database is **PostgreSQL** (not SQLite); a failed final save is **returned** on the result (`persistence_error`), not raised.

- **`persistence.py`**: the `PersistenceStore` protocol and what a chat UI reads: `ChatRecord` (owner, title, status, metadata, counts, tokens, version, fork origin, timestamps, archive/delete, lease), `ChatPage` (opaque cursor), `StoredMessage`, `RunRecord`, `Lease`. Plus `fork_state` / `fork_chat` (edit a message or regenerate an answer = a new chat from a prefix ending before a user message) and `RunPersistence`, which the runtime uses during a run.
- **When the runtime saves** (`Runtime(store=..., lease_seconds=...)`, then `run(..., chat_id=, owner_id=, metadata=)` / `resume(..., chat_id=)`):
  - after preparing the run, before the model is called (the person's message is never lost),
  - after every step that leads to another (`LoopDependencies.checkpoint`, called by the loop),
  - at the end whatever the status (completed, awaiting_input, failed, cancelled, timed_out, max_steps), releasing the lease.
  Events are buffered by `RunPersistence` (an observer) and written in the same transaction as the state.
- **Deviation from the plan: the runtime saves, not a `PersistenceObserver`.** Observers are best-effort (timeouts and failures are swallowed), which suits events but not "the answer was saved". Mid-run checkpoint failures are logged and the run carries on; a lost lease stops the run; a failed final save is returned as `RunResult.persistence_error`.
- **One run per chat**: `acquire` takes a lease (token + expiry), renewed every third of `lease_seconds` by a heartbeat. A second request is refused with `PersistenceConflictError` before anything runs (double-clicked Approve runs the tool once). Every save checks the lease token. A state older than the store (fewer or different messages) is refused instead of overwriting newer ones.
- **Crash recovery**: a chat stored as `running` whose lease expired is recovered by the next run: `_repair_transcript` closes open tool calls with `RUN_INTERRUPTED` ("its outcome is unknown; it was not retried"), held results keep their real values, `RunRecord.recovered` is set.
- **Storage layout**: messages are append-only, so each save inserts only the new ones; the rest of the state is one snapshot (`state` JSONB, without messages). Load = snapshot + messages in order.
- **`PostgresStore`** (`stores/postgres.py`, asyncpg, optional extra `agent-harness[postgres]`): tables `harness_chats`, `harness_messages`, `harness_runs`, `harness_events` (+ `harness_schema_migrations`), configurable schema and prefix, versioned migrations applied on first use under an advisory lock, indexes for "an owner's chats, newest first", status, metadata (GIN) and forks. Lease expiry uses the database clock, so several app servers can share it.
- **`InMemoryStore`** (`stores/memory.py`): the reference implementation, copies in and out like a database.
- **Errors**: `PersistenceError` / `PersistenceConflictError`, codes `PERSISTENCE_ERROR`, `PERSISTENCE_CONFLICT`, `RUN_INTERRUPTED`. Event `checkpoint.saved` (transient).
- **CLI** (`main.py`): `HARNESS_DATABASE_URL`, `HARNESS_DB_SCHEMA`, `HARNESS_LEASE_SECONDS`, `HARNESS_OWNER_ID`; `--owner`, `--chat`, `--list`, `--show`, `--resume <chat id>`. Without a database the session file is now rewritten after every run (it used to keep the suspended state after the run had completed).
- **No tests written** (explicit instruction, which replaces the planned contract suite); verified with a scratch script running the same scenarios against both stores (a throwaway PostgreSQL 14): save/load round trip, continue by id, ask → answer → reload shows completed, concurrent request refused, double approve runs once, heartbeat keeps a long run's lease, crash mid-tool recovered without re-running, failed final save returned, list/paginate/filter/rename/archive/delete, message paging, fork, stale state refused, migrations idempotent. Real Gemini runs through the CLI: question answered, `--resume` of the finished chat shows it completed with its answer, `--list`, `--chat` continue, `--show`.

## Phase 10: Compaction

**Build:**
- The `Compactor` protocol, and `PruneToolResultsCompactor` (model-free).
- `SummarizingCompactor`:
  - selects a region, keeping a recent tail and tool pairing;
  - summarizes using the provider;
  - stores a `CompactionCheckpoint` in `state.extensions`, which the `ContextManager` applies as a view.
- Triggered by the budget, and on `CONTEXT_WINDOW_EXCEEDED` followed by a single retry.

### Phase 10 outcome

- **Use case that shaped it:** long debugging / incident-RCA sessions that gather data from many sources and must not forget older findings. Artifacts (Phase 7) handle *one* oversized result; compaction handles a conversation that has *slowly* grown too big.
- **Implemented** in `compaction.py` as one concrete `Compactor` class (no protocol, no model-free `PruneToolResultsCompactor`: pruning old tool results is what Phase 7 offloading and Phase 3 dropping already do).
- **Region:** history is split into first message (always kept) / region (older middle, summarized) / recent tail (kept verbatim, `compaction_keep_recent_ratio=0.3` of the budget). The tail starts on a segment boundary (`context._segment_history`), so a tool call is never separated from its results.
- **Trigger:** `compaction_trigger_ratio=0.8` of `max_context_tokens - reserved_output_tokens`, checked in `loop.py` before each LLM call. Size is measured from the provider's reported `input_tokens` of the previous request plus a chars/4 estimate of only the messages added since (`usage_anchor`); falls back to a full estimate when the provider reports no usage, after a compaction, or when Phase 3 dropped messages from that request.
- **Summary:** one extra LLM call, by default to the chat provider and `agent.model` (`Compactor(llm=..., model=...)` to change). The prompt asks for fixed sections (user requests verbatim, timeline, evidence with identifiers copied exactly, sources checked, hypotheses, conclusions, open questions). Each later compaction gets the previous summary and folds it in (rolling). Messages over 24,000 characters are middle-truncated in the summarizer's input only.
- **Archive (decision: archive each region):** before a region is replaced, its original messages are saved word for word to the `ArtifactStore` as `compacted_region_<n>_<id>`, with a header (message range, tools called, size). The harness, not the model, appends an "Archives" index to the summary message listing every archive, so the model always knows what exists and can search it with `read_artifact`. The same index is kept in `state.extensions["compaction"]["archives"]`, plus a per-compaction `history` log.
- **History is never edited (D3):** the checkpoint (`summary`, `covered_until`, `count`, `archives`) lives in `state.extensions["compaction"]`; `compacted_view(state)` applies it and `DefaultContextManager.build` uses it, then Phase 3 `_fit_budget` still applies on top as the safety net.
- **Provider overflow:** a `CONTEXT_WINDOW_EXCEEDED` error from the provider triggers one forced compaction and one retry of that step (works even without `max_context_tokens`). Without a compactor, or if nothing can be compacted, the original error propagates.
- **Failure handling:** if the summary call fails or comes back empty, nothing changes, Phase 3 dropping keeps the request in budget, and compaction isn't retried for 3 steps. A failed archive write is logged and the summary is still used.
- **Opt-in:** `Runtime(compactor=...)`, same pattern as `hooks`/`result_processor`.
- **main.py:** `HARNESS_COMPACTION` (default on), `HARNESS_COMPACTION_TRIGGER`, `HARNESS_COMPACTION_KEEP_RECENT`, `HARNESS_COMPACTION_MODEL`, `HARNESS_COMPACTION_SUMMARY_MAX_TOKENS`. One `FileArtifactStore` and one `read_artifact` tool are shared by offloading and archives.
- **No tests written** (explicit instruction); verified with a scripted fake provider: repeated compactions produce contiguous, non-overlapping archives starting at message 1 and a rolling summary that sees the previous one; every request stayed in budget with tool pairs intact; a failing summarizer falls back to dropping with a cooldown; provider overflow compacts and retries (and still fails without a compactor); measured usage triggers compaction; the checkpoint survives multi-turn chat.

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

### Phase 12 outcome

- **Implemented** `ParallelToolExecutor` in `executor.py`, a subclass of `SequentialToolExecutor`. Both now share two stages: `_prepare` (lookup, validation, `before_tool`) and `_invoke` (tool body + timeout). The sequential executor behaves exactly as before.
- **Per batch (one model response):** (1) every call is prepared in model order before any body starts, so policy hooks never run concurrently and a denied call simply has its result already; (2) consecutive `concurrency_safe` calls run together under an `asyncio.Semaphore(agent.limits.max_parallel_tool_calls)` (default 10, a sliding pool rather than chunks), while a call that isn't concurrency-safe is a barrier that runs alone after everything before it and before everything after it; (3) `after_tool` runs and results are returned in model order.
- **Failure isolation:** `_invoke` never raises (except cancellation), so one failure or timeout only affects its own result. Cancelling the run cancels every running call (via `asyncio.TaskGroup`), and transcript repair answers them as before.
- **Across runs:** `ParallelToolExecutor(max_total=N)` adds an optional semaphore shared by every run using that executor (e.g. many chats on one server). Each run's own batch limit still applies.
- **Dependent calls** need nothing special: the model only issues calls whose arguments it already knows, so "look up X, then fetch A/B/C for X" naturally becomes one step with one call, then one step with three parallel calls.
- **`gather_agents` not built:** subagents don't exist until Phase 11; it will be added there.
- **main.py:** `HARNESS_PARALLEL_TOOLS` (default true), `HARNESS_MAX_PARALLEL_TOOLS` (10), `HARNESS_MAX_PARALLEL_TOOLS_TOTAL` (blank = no cross-run cap). `INSTRUCTIONS` now asks the model to request independent tool calls in the same response. `read_artifact` is marked `concurrency_safe`; the app's tools already were.
- **No tests written** (explicit instruction). Verified with a scripted provider and timed tool bodies: 4 safe reads ran together (0.4s instead of 1.0s) and came back in model order; an unsafe call waited for the earlier reads and the later reads waited for it; a cap of 3 held with 8 calls; a raising call and a timed-out call didn't affect their neighbours; `before_tool` hooks all ran before any body and `after_tool` ran in order; cancel mid-batch answered every call; a lookup-then-fan-out sequence ran as 1 call then 3 parallel calls; `max_total=3` held across two concurrent runs; the sequential executor was unchanged. With the real model (Gemini), "weather in Paris, Tokyo and London" ran as one step with 3 parallel `get_weather` calls.

## Phase 12b: Human-in-the-Loop (HITL)

*Added 2026-11-06 at the user's request, replacing the Phase 0 decision to defer HITL. To be planned in detail (and reviewed) after Phase 12.*

**Scope (what the user asked for):**
- **Approve / edit / reject** a tool call before it runs. Edit means the person changes the arguments. Reject comes with a reason the model sees, so it can adjust.
- **Forms:** the harness, or the model through a tool, asks the person a structured question with options to pick (single or multi-select).
- **Free-text answer:** the person can answer in their own words instead of picking an option ("Other").
- **Notes:** the person can attach a note to any response (approve, edit, reject or a form answer), which the model also sees.

**Questions to settle when planning:**
- Who triggers a request: policy hooks (`before_tool` returning `ask`, or rules on `destructive` annotations) and/or the model through an `ask_user` tool.
- How the harness waits: an in-process async `InteractionHandler` (the run pauses while connected, e.g. CLI prompt or web UI) and/or suspend-and-resume (the run ends as `AWAITING_INPUT` with a serializable pending request and resumes later with the answer). The second needs the resumable state from Phase 1.
- Timeouts and defaults (what happens if nobody answers), fail-closed behavior, and events (`interaction.requested` / `interaction.answered`) for the UI.
- How parallel tool calls from Phase 12 interact: policy and approval stages run serially in model order, before tool bodies run concurrently.

### Phase 12b outcome

- **Decision: suspend-and-resume only** (no in-memory waiting). When input is needed the run ends with `RunStatus.AWAITING_INPUT`; the app saves `state.model_dump_json()` wherever it likes and later calls `runtime.resume(agent, state, answers=[...])`. Nothing is held in memory, so it survives restarts and long waits. Phase 9 will only automate the saving.
- **`hitl.py`:** `Interaction` (the open request: kind `approval`/`input`, call id, tool, arguments, message, options, `multi_select`, `allow_free_text`, `allow_edit`), `Answer` with helpers `approve`/`edit`/`reject`/`answer` (each takes an optional `note` and `answered_by`), `InputRequired` (raised by a tool to ask), `make_ask_user_tool()`, and the state helpers. Everything lives in `state.extensions["hitl"]`: `pending`, `answers`, `held` (finished results of the suspended batch), `history` (audit trail).
- **Two triggers:** a `before_tool` hook returns `ask(reason, allow_edit=True)` -> approval (approve / edit arguments / reject); a tool raises `InputRequired(question, options=...)` -> input (answer / reject). `ask_user` is the built-in tool for the second; on resume the tool runs again with `ctx.answer` set.
- **Executor:** on resume, `reject` returns a `TOOL_REJECTED` result with the note (the tool never runs); `edit` validates the new arguments (invalid -> error result to the model) and hooks see the edited call; an approved call that a hook asks about again is allowed, but a `deny` still wins. The model sees "[The user approved / edited ... User's note: ...]" appended to the result. Calls waiting for input return an `AWAITING_INPUT` result (the `ToolExecutor` contract is unchanged), skip `after_tool`, and emit `tool.awaiting_input`.
- **Batches:** calls that need no input still run before the run suspends; their results are held and committed with the rest, in model order, once everything is answered (never re-run). A call that is not concurrency-safe and comes after one waiting for input is deferred (`tool.deferred`) and runs on resume. Several requests from one batch are asked together; answering only some suspends again with the rest.
- **Loop/runtime:** `_execute_tools` returns whether the run suspended; `runtime.resume()`, `answers=` on `start/run/run_sync`, `RunResult.awaiting_input` and `.pending_interactions`. New input while waiting is refused. Transcript repair (cancel/fail during a resume) keeps held results and clears pending requests.
- **Events:** `interaction.requested` (the full request, for the UI), `interaction.resolved`, `tool.awaiting_input`, `tool.deferred`, `step.completed(outcome=awaiting_input)`, `run.completed(status=awaiting_input, pending_interactions=[...])`. Event `seq` now continues across resumes (stored in `state.extensions["events"]`).
- **Timeouts:** the harness doesn't wait, so it has no timer; an app that wants "no answer in N hours = reject" resumes with `reject(...)` itself.
- **main.py:** `HARNESS_ASK_USER` (default true), `HARNESS_APPROVAL_TOOLS` (comma list), `HARNESS_SESSIONS_DIR` (default `.sessions/`); the CLI prompts for answers, `later` exits with the run saved, `python main.py --resume <file>` continues it in a new process. The CLI's own `ask()` was renamed `run_turn()` (it shadowed the hook helper).
- **No tests written** (explicit instruction). Verified with a scripted provider for both executors (approve, edit, invalid edit, reject with note, `ask_user` with invalid-answer guards, mixed batch with held/deferred calls, partial answers, deny-over-approval, API guards, cancel mid-resume keeping held results), every state round-tripped through JSON, and with Gemini: approve-with-edit in one session, and an `ask_user` question left with `later` and answered from a new process via `--resume`.

## Phase 13: MCP

**Build:**
- `agent_harness_ext.mcp`: an `MCPToolSource` that connects over stdio or streamable HTTP, lists the server's tools, wraps each one as a `Tool` named `mcp__server__tool`, and has timeouts and reconnect.
- Optional deferred loading through meta-tools.
- `mcp` is an optional extra.

## Phase 14: Safety and Governance

**Build:**
- A `ToolPolicy`: allow and deny lists, and rules based on annotations such as "approve `destructive`".
- ~~`ApprovalHandler` protocol~~ and the `AWAITING_INPUT` suspend-and-resume path: **moved to Phase 12b (HITL)**.
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
