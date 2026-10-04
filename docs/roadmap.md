# Roadmap

| Phase | Name | Status | Needed? (re-evaluate at each gate) |
|---|---|---|---|
| 0 | Repository analysis, architecture, plan | **Done — reviewed** | — |
| 1 | Core agent harness | **Done — awaiting review** (91 tests passing) | Yes |
| 2 | Reliability & execution control | **Done** (retry/timeout middleware, cancellation, transcript repair, repeated-call guard) | Yes |
| 3 | Context management | **Done** (token budget, ordered prompt sections, oldest-group reduction, overflow) | Yes |
| 4 | Tool execution pipeline (hooks) | Not started | Yes |
| 5 | Event system | Not started | Yes |
| 6 | Observability | Not started | Yes |
| 7 | Large result handling | Not started | Likely |
| 8 | Streaming | Not started | Likely |
| 9 | Persistence | Not started | Likely |
| 10 | Compaction | Not started | Evaluate after 3 and 7 |
| 11 | Subagents | Not started | Evaluate |
| 12 | Parallel execution | Not started | Evaluate |
| 13 | MCP | Not started | Evaluate |
| 14 | Safety & governance | Not started | Evaluate |
| 15 | Replay | Not started | Evaluate after 9 |
| 16 | Plugin architecture | Not started | Formalize what exists |

## Phase 0 review decisions

| # | Question | Decision |
|---|---|---|
| 1 | Package name | **`agent_harness`** (distribution name `agent-harness`) |
| 2 | Source of truth / storage | **In-memory `AgentState` is the source of truth during a run.** No database in the core. Phase 9 adds a `PersistenceStore` interface (in-memory + optional SQLite/Postgres extra) modeled on TrueForge: an immutable message log plus per-run message-id arrays, save-before-mutate at step boundaries, and a shared store contract test suite. The context-engineering techniques (offload, compaction) come from TrueForge in Phases 3, 7 and 10. |
| 3 | `finish_reason=length` handling | **Continue up to 3 times** (`AgentLimits.max_truncation_continuations`). Then fail with `OUTPUT_TRUNCATED`, keeping the partial text. Implemented in Phase 1. |
| 4 | Human-in-the-loop | **Not required now.** Approval and suspend/resume are deferred. Phase 14 keeps only allow/deny policy. |
| 5 | Default `max_steps` | **15** |
