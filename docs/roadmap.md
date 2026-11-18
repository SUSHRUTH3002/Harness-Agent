# Roadmap

| Phase | Name | Status | Needed? (re-evaluate at each gate) |
|---|---|---|---|
| 0 | Repository analysis, architecture, plan | **Done — reviewed** | — |
| 1 | Core agent harness | **Done — awaiting review** (91 tests passing) | Yes |
| 2 | Reliability & execution control | **Done** (retry/timeout middleware, cancellation, transcript repair, repeated-call guard) | Yes |
| 3 | Context management | **Done** (token budget, ordered prompt sections, oldest-group reduction, overflow) | Yes |
| 4 | Tool execution pipeline (hooks) | **Done** (`before_step`, `before_tool`, `after_tool`, `before_stop` -- `wrap_llm`/`transform_request`/`wrap_tool` served by existing composition, not a second hook list) | Yes |
| 5 | Event system | **Done** (`EventBus` + per-run observers, open dotted event types, `emit_event` from anywhere in a run via a context variable, ordered `seq`, isolated observers, `JsonlEventWriter` / `PrintObserver`) | Yes |
| 6 | Observability | Not started | Yes |
| 7 | Large result handling | **Done** (offload oversized/combined results to an `ArtifactStore` + `read_artifact`; no "summarize" strategy, see outcome notes) | Likely |
| 8 | Streaming | **Done** (`StreamingLLMProvider.stream()` with neutral `LLMStreamEvent`s for LiteLLM and local servers; one `StreamAssembler` builds the same `LLMResponse` as before; transient `llm.token` / `llm.tool_call_delta` / `llm.stream_reset` events, never saved; retry restarts a failed stream with a reset; idle timeout after the first piece; `Execution.events()` for live reading; CLI prints as the model writes, `--no-stream`) | Yes (a chat UI needs it) |
| 9 | Persistence | **Done** (`PersistenceStore` with `InMemoryStore` and `PostgresStore`; the runtime saves at start, after each continuing step and at the end whatever the status; chats load by id and list by owner (user, incident, ...); a lease allows one run per chat and recovers crashed runs without re-running their tool calls; fork for edit/regenerate) | Yes |
| 10 | Compaction | **Done** (80% trigger, rolling evidence-preserving summary, each region archived to the `ArtifactStore`, real-usage measurement, compact-and-retry on provider overflow) | Yes (long RCA/debugging sessions) |
| 11 | Subagents | Not started | Evaluate |
| 12 | Parallel execution | **Done** (`ParallelToolExecutor`: concurrency-safe calls run together in a sliding pool, unsafe calls are barriers, policy and results in model order, per-call failure isolation, optional cross-run cap) | Yes (RCA pulls from many sources at once) |
| 12b | Human-in-the-loop (HITL) | **Done** (suspend-and-resume: the run ends `AWAITING_INPUT` with everything in serializable state, resumes with answers even after a restart; approve / edit / reject via `before_tool` `ask`, `ask_user` forms with options or free text, notes on every answer) | Yes |
| 13 | MCP | Not started | Evaluate |
| 14 | Safety & governance | Not started | Evaluate |
| 15 | Replay | Not started | Evaluate after 9 |
| 16 | Plugin architecture | Not started | Formalize what exists |
| 17 | Chat API and Web UI | **17a Chat API done** (FastAPI + WebSocket over `Runtime`/`PersistenceStore`: chats, messages, HITL answers, cancel, live replayable run stream, bearer token, owner scoping); 17b Web UI designed, not started. Called "Phase 11a/11b" in conversation; Phase 11 stays Subagents | Yes |

## Phase 0 review decisions

| # | Question | Decision |
|---|---|---|
| 1 | Package name | **`agent_harness`** (distribution name `agent-harness`) |
| 2 | Source of truth / storage | **In-memory `AgentState` is the source of truth during a run.** No database in the core. Phase 9 adds a `PersistenceStore` interface (in-memory + optional SQLite/Postgres extra) modeled on TrueForge: an immutable message log plus per-run message-id arrays, save-before-mutate at step boundaries, and a shared store contract test suite. The context-engineering techniques (offload, compaction) come from TrueForge in Phases 3, 7 and 10. |
| 3 | `finish_reason=length` handling | **Continue up to 3 times** (`AgentLimits.max_truncation_continuations`). Then fail with `OUTPUT_TRUNCATED`, keeping the partial text. Implemented in Phase 1. |
| 4 | Human-in-the-loop | **Not required now.** Approval and suspend/resume are deferred. Phase 14 keeps only allow/deny policy. *Superseded 2026-11-06: HITL is now planned as Phase 12b.* |
| 5 | Default `max_steps` | **15** |
