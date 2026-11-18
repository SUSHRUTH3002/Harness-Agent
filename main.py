"""Run the agent harness against a real model, configured from .env.

    python main.py "What's the weather in Tokyo in Fahrenheit?"   # one question
    python main.py                                                # interactive chat
    python main.py -q "..."                                       # answer only, no transcript
    python main.py --no-stream "..."                              # print the answer when it is complete

In a terminal the answer is printed as the model writes it (HARNESS_STREAM=true, the default).

With HARNESS_DATABASE_URL set, every chat is saved in PostgreSQL:

    python main.py --owner INC-1001 "Why is db-3 slow?"   # new chat, owned by an incident (or a user id)
    python main.py --list --owner INC-1001                 # that owner's chats, newest first
    python main.py --chat <id> "and the replica?"          # continue a chat
    python main.py --chat <id>                             # reopen it interactively
    python main.py --show <id>                             # print its conversation
    python main.py --resume <id>                           # answer what it is waiting for

Configuration lives in .env (see .env.example): HARNESS_MODEL plus the key
for that provider, e.g. ANTHROPIC_API_KEY or OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import litellm
from dotenv import dotenv_values

from agent_harness import (
    Agent,
    AgentLimits,
    AgentState,
    Answer,
    Compactor,
    EventBus,
    EventType,
    Execution,
    FileArtifactStore,
    HarnessError,
    Hooks,
    Interaction,
    JsonlEventWriter,
    LargeResultProcessor,
    ParallelToolExecutor,
    PrintObserver,
    ResilientLLMProvider,
    RetryPolicy,
    Role,
    RunResult,
    RunStatus,
    Runtime,
    allow,
    answer,
    approve,
    ask,
    deny,
    edit,
    make_ask_user_tool,
    make_read_artifact_tool,
    reject,
)
from agent_harness import pending_interactions as pending_interactions_of
from agent_harness.messages import Message, new_id
from agent_harness.providers.litellm_provider import LiteLLMProvider
from agent_harness.providers.local_provider import LocalProvider
from agent_harness.testing import calculator, get_weather, web_search
from agent_harness.tools_dir.agent_tools import fetch_resource_metadata, fetch_resource_neighbors

ROOT = Path(__file__).resolve().parent
# LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Each of these attaches its own handler directly to its own logger (not just the root), so a
# quiet root level alone does not silence it — e.g. litellm's "LiteLLM" logger
# (litellm/_logging.py) keeps emitting through that handler regardless of root's level unless
# capped here explicitly. Capped at WARNING regardless of HARNESS_LOG_LEVEL / -v.
_NOISY_LIBRARIES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "httpcore", "openai", "instructor")

logger = logging.getLogger("main")

# LiteLLM prints help banners to stderr on errors; the harness already reports errors itself.
litellm.suppress_debug_info = False

INSTRUCTIONS = (
    "You are a helpful assistant. Use the available tools for weather, arithmetic and search "
    "instead of guessing. When you need several independent pieces of information, request all "
    "of those tool calls in the same response so they can run in parallel. If you need a decision or "
    "information only the user has, use ask_user instead of guessing. Answer concisely."
)

# Banned search terms for the web_search tool. Queries containing these terms will be blocked.
_BANNED_SEARCH_TERMS = ("bomb", "weapon")


def build_hooks(approval_tools: frozenset[str] = frozenset()) -> Hooks:
    hooks = Hooks()

    # Requires approval for certain tools as specified in HARNESS_APPROVAL_TOOLS.
    @hooks.on_before_tool
    def require_approval(call, ctx):
        if call.name in approval_tools:
            return ask(f"'{call.name}' is configured to need approval (HARNESS_APPROVAL_TOOLS).")
        return allow()

    @hooks.on_before_tool
    def block_banned_search_terms(call, ctx):
        if call.name != "web_search":
            return allow()
        query = str((call.arguments or {}).get("query", "")).lower()
        hit = next((term for term in _BANNED_SEARCH_TERMS if term in query), None)
        if hit:
            return deny(f"web_search queries containing '{hit}' are blocked by policy.")
        return allow()

    @hooks.on_before_tool
    def limit_resource_batch_size(call, ctx):
        if call.name == "fetch_resource_metadata":
            ids = call.arguments.get("resource_ids", [])
            if len(ids) > 5:
                return deny(f"requested {len(ids)} resources at once; limit is 5.")
        return allow()

    return hooks


class ConfigError(Exception):
    pass


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _number(name: str, kind: type, default=None):
    raw = _env(name)
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a {kind.__name__}, got {raw!r}") from exc


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    if raw.strip().lower() not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_env_file(path: Path) -> None:
    """Export non-empty .env entries. Shell variables win, so `HARNESS_MODEL=... python main.py` works.

    Blank entries (e.g. an unfilled `OPENAI_API_KEY=` from the template) must not exist as empty
    variables, or key checks see a key that isn't there. LiteLLM runs its own `load_dotenv()` at
    import time, which exports those blanks, so they are removed here as well.
    """
    for key, value in dotenv_values(path).items():
        value = (value or "").strip()
        current = os.environ.get(key)
        if value:
            if not current:
                os.environ[key] = value
        elif current is not None and not current.strip():
            del os.environ[key]


def setup_logging(level_name: str, log_file: str | None) -> None:
    """Harness and app loggers at `level_name`; noisy third-party libraries stay at WARNING."""
    level = logging.getLevelName(level_name.upper())
    if not isinstance(level, int):
        raise ConfigError(f"HARNESS_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {level_name!r}")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True)
    for name in ("agent_harness", "main"):
        logging.getLogger(name).setLevel(level)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def load_config() -> dict:
    load_env_file(ROOT / ".env")
    
    local = _bool("HARNESS_LOCAL_MODEL")
    model = _env("HARNESS_MODEL")
    if not model:
        hint = "e.g. llama-3.1-8b-instruct" if local else "e.g. openai/<model>"
        raise ConfigError(f"HARNESS_MODEL is not set. Copy .env.example to .env and set it, {hint}.")
    api_base, api_key = _env("HARNESS_API_BASE"), _env("HARNESS_API_KEY")

    if local:
        if not api_base:
            raise ConfigError(
                "HARNESS_LOCAL_MODEL=true requires HARNESS_API_BASE, e.g. http://localhost:11434/v1 (Ollama)."
            )
    elif not (api_base or api_key):
        missing = litellm.validate_environment(model).get("missing_keys") or []
        if missing:
            raise ConfigError(f"model {model!r} needs {', '.join(missing)} in .env (or HARNESS_API_BASE/HARNESS_API_KEY).")

    settings = {}
    # Left unset by default rather than forced to 0: some models (e.g. Gemini 3) warn or
    # degrade when pushed below their own recommended default.
    temperature = _number("HARNESS_TEMPERATURE", float)
    if temperature is not None:
        settings["temperature"] = temperature
    max_tokens = _number("HARNESS_MAX_TOKENS", int)
    if max_tokens:
        settings["max_tokens"] = max_tokens

    repeat_action = _env("HARNESS_REPEAT_CALL_ACTION") or "warn"
    if repeat_action not in ("warn", "stop"):
        raise ConfigError(f"HARNESS_REPEAT_CALL_ACTION must be 'warn' or 'stop', got {repeat_action!r}")

    return {
        # Phase 7: large result handling
        "large_result_handling": _bool("HARNESS_LARGE_RESULT_HANDLING", True),
        "large_result_per_call_tokens": _number("HARNESS_LARGE_RESULT_PER_CALL_TOKENS", int, 6000),
        "large_result_per_step_tokens": _number("HARNESS_LARGE_RESULT_PER_STEP_TOKENS", int, 10000),
        "large_result_preview_chars": _number("HARNESS_LARGE_RESULT_PREVIEW_CHARS", int, 100),
        "artifact_dir": _env("HARNESS_ARTIFACT_DIR") or str(ROOT / ".artifacts"),
        "artifact_read_max_chars": _number("HARNESS_ARTIFACT_READ_MAX_CHARS", int, 4000),
        # Phase 10: compaction
        "compaction": _bool("HARNESS_COMPACTION", True),
        "compaction_trigger_ratio": _number("HARNESS_COMPACTION_TRIGGER", float, 0.8),
        "compaction_keep_recent_ratio": _number("HARNESS_COMPACTION_KEEP_RECENT", float, 0.3),
        "compaction_model": _env("HARNESS_COMPACTION_MODEL"),
        "compaction_summary_max_tokens": _number("HARNESS_COMPACTION_SUMMARY_MAX_TOKENS", int, 4096),
        # Phase 12: parallel tool execution
        "parallel_tools": _bool("HARNESS_PARALLEL_TOOLS", True),
        "max_parallel_tools": _number("HARNESS_MAX_PARALLEL_TOOLS", int, 10),
        "max_parallel_tools_total": _number("HARNESS_MAX_PARALLEL_TOOLS_TOTAL", int),
        # Phase 12b: human-in-the-loop
        "ask_user": _bool("HARNESS_ASK_USER", True),
        "approval_tools": frozenset(t.strip() for t in (_env("HARNESS_APPROVAL_TOOLS") or "").split(",") if t.strip()),
        "sessions_dir": _env("HARNESS_SESSIONS_DIR") or str(ROOT / ".sessions"),
        # Phase 9: persistence
        "database_url": _env("HARNESS_DATABASE_URL"),
        "db_schema": _env("HARNESS_DB_SCHEMA") or "public",
        "lease_seconds": _number("HARNESS_LEASE_SECONDS", float, 600.0),
        "owner_id": _env("HARNESS_OWNER_ID"),
        # Phase 8: streaming
        "stream": _bool("HARNESS_STREAM", True),
        "stream_idle_timeout": _number("HARNESS_STREAM_IDLE_TIMEOUT", float, 30.0),
        # Phase 5: events
        "events_file": _env("HARNESS_EVENTS_FILE"),
        "print_events": _bool("HARNESS_PRINT_EVENTS"),
        "model": model,
        "local": local,
        "api_base": api_base,
        "api_key": api_key,
        "timeout": _number("HARNESS_TIMEOUT", float, 120.0),
        "max_steps": _number("HARNESS_MAX_STEPS", int, 15),
        "settings": settings,
        "max_retries": _number("HARNESS_MAX_RETRIES", int, 3),
        "max_execution_time": _number("HARNESS_MAX_EXECUTION_TIME", float),
        "repeat_call_threshold": _number("HARNESS_REPEAT_CALL_THRESHOLD", int, 3),
        "repeat_call_action": repeat_action,
        "max_context_tokens": _number("HARNESS_MAX_CONTEXT_TOKENS", int),
        "reserved_output_tokens": _number("HARNESS_RESERVED_OUTPUT_TOKENS", int, 1024),
    }


def build(config: dict) -> tuple[Runtime, Agent]:
    if config["local"]:
        provider = LocalProvider(
            config["model"], base_url=config["api_base"], api_key=config["api_key"] or "not-needed",
            timeout=config["timeout"],
        )
    else:
        provider = LiteLLMProvider(
            config["model"], api_base=config["api_base"], api_key=config["api_key"], timeout=config["timeout"]
        )
    
    # Retries live outside the loop entirely, this wrapper is the only thing that changes when retry behavior changes. (refer retry.py)
    # idle_timeout (Phase 8): a stream that goes quiet that long after it started is retried.
    provider = ResilientLLMProvider(provider, policy=RetryPolicy(max_retries=config["max_retries"]),
                                    idle_timeout=config["stream_idle_timeout"])

    tools = [calculator, get_weather, web_search, fetch_resource_metadata, fetch_resource_neighbors]
    if config["ask_user"]:
        tools.append(make_ask_user_tool())
    result_processor = None
    compactor = None
    if config["large_result_handling"] or config["compaction"]:
        # One store for both offloaded results (Phase 7) and compaction archives (Phase 10), so a
        # single read_artifact tool reads either. File-backed so they survive past this process and
        # are inspectable on disk; swap for InMemoryArtifactStore() if you don't want files left behind.
        store = FileArtifactStore(config["artifact_dir"])
        tools.append(make_read_artifact_tool(store, max_chars=config["artifact_read_max_chars"]))
        if config["large_result_handling"]:
            result_processor = LargeResultProcessor(store=store)
        if config["compaction"]:
            # Summaries use the chat provider and model unless HARNESS_COMPACTION_MODEL is set
            # (any LiteLLM model string; in local mode, another model on the same server).
            compactor = Compactor(store=store, model=config["compaction_model"],
                                  summary_max_tokens=config["compaction_summary_max_tokens"])

    agent = Agent(
        name="assistant",
        instructions=INSTRUCTIONS,
        tools=tools,
        model_settings=config["settings"],
        limits=AgentLimits(
            max_steps=config["max_steps"],
            max_execution_time=config["max_execution_time"],
            llm_timeout=config["timeout"],
            repeat_call_threshold=config["repeat_call_threshold"],
            repeat_call_action=config["repeat_call_action"],
            max_context_tokens=config["max_context_tokens"],
            reserved_output_tokens=config["reserved_output_tokens"],
            large_result_per_call_tokens=config["large_result_per_call_tokens"],
            large_result_per_step_tokens=config["large_result_per_step_tokens"],
            large_result_preview_chars=config["large_result_preview_chars"],
            compaction_trigger_ratio=config["compaction_trigger_ratio"],
            compaction_keep_recent_ratio=config["compaction_keep_recent_ratio"],
            max_parallel_tool_calls=config["max_parallel_tools"],
        ),
    )
    hooks = build_hooks(config["approval_tools"])
    # Concurrency-safe tools in one model response run at the same time; the rest stay in order.
    # A custom executor must be given the hooks itself (Runtime only wires them into its default).
    executor = (
        ParallelToolExecutor(hooks=hooks, max_total=config["max_parallel_tools_total"])
        if config["parallel_tools"] else None
    )
    # Observers here see every run; an app can also pass observers=[...] to runtime.run() for one run.
    events = EventBus()
    if config["events_file"]:
        events.subscribe(JsonlEventWriter(config["events_file"]))
    if config["print_events"]:
        events.subscribe(PrintObserver())
    # Phase 9: with a database, every run loads and saves its chat there. Without one, runs that wait
    # for input are kept in session files instead (HARNESS_SESSIONS_DIR).
    store = None
    if config["database_url"]:
        from agent_harness.stores.postgres import PostgresStore  # needs asyncpg, so only when configured

        store = PostgresStore(config["database_url"], schema=config["db_schema"])
    runtime = Runtime(provider, hooks=hooks, tool_executor=executor, result_processor=result_processor,
                      compactor=compactor, events=events, store=store, lease_seconds=config["lease_seconds"],
                      stream=config["stream"])
    return runtime, agent


def print_transcript(result: RunResult, start: int) -> None:
    for message in result.state.messages[start:]:
        if message.role is Role.TOOL:
            tag = f"tool:{message.name}" + (" (error)" if message.is_error else "")
            print(f"  [{tag}] {message.text[:300]}")
        elif message.role is Role.ASSISTANT:
            for c in message.tool_calls:
                args = json.dumps(c.arguments) if c.arguments is not None else c.raw_arguments
                print(f"  [call -> {c.name}] {args}")


def print_summary(result: RunResult) -> None:
    cost = f" | ${result.usage.cost_usd:.6f}" if result.usage.cost_usd is not None else ""
    print(
        f"  ({result.status.value} | {result.steps} step(s) | {result.duration:.2f}s | "
        f"tokens in={result.usage.input_tokens} out={result.usage.output_tokens}{cost})"
    )


def print_history(messages: list[Message]) -> None:
    """A saved conversation, the way a chat window shows it."""
    for message in messages:
        if message.role is Role.USER:
            print(f"\nyou> {message.text}")
        elif message.role is Role.ASSISTANT:
            for c in message.tool_calls:
                args = json.dumps(c.arguments) if c.arguments is not None else c.raw_arguments
                print(f"  [call -> {c.name}] {args}")
            if message.text:
                print(f"\n{message.text}")
        elif message.role is Role.TOOL:
            tag = f"tool:{message.name}" + (" (error)" if message.is_error else "")
            print(f"  [{tag}] {message.text[:300]}")


async def print_live(execution: Execution) -> bool:
    """Show the run as it happens: the model's words as it writes them, and each tool call and
    result as it finishes. Returns True if any answer text was shown."""
    shown = False  # text shown for the response being written now
    any_text = False
    async for event in execution.events():
        if event.type == EventType.LLM_TOKEN:
            if not shown:
                print()
            print(event.data["text"], end="", flush=True)
            shown = any_text = True
        elif event.type == EventType.LLM_STREAM_RESET:
            # A terminal can't take back printed text, so say what happened.
            if shown:
                print("\n  [connection dropped; retrying. Ignore the partial text above]", flush=True)
            shown = False
        elif event.type == EventType.LLM_COMPLETED:
            if shown:
                print(flush=True)
            shown = False
        elif event.type == EventType.TOOL_STARTED:
            print(f"  [call -> {event.data['tool']}] {json.dumps(event.data.get('arguments'))}", flush=True)
        elif event.type == EventType.TOOL_COMPLETED:
            tag = f"tool:{event.data['tool']}" + ("" if event.data["ok"] else " (error)")
            print(f"  [{tag}] {(event.data.get('content') or '')[:300]}", flush=True)
    return any_text


async def run_turn(
    runtime: Runtime, agent: Agent, question: str | None, state: AgentState | None, quiet: bool,
    sessions_dir: Path, answers: list[Answer] | None = None, *, chat_id: str | None = None,
    owner_id: str | None = None, session_path: Path | None = None,
) -> RunResult:
    """Run (or resume) until the run finishes or the person chooses to answer later.

    With a database the runtime saves the chat after every run by itself (`chat_id`). Without one,
    a run that ever waited for input is kept in a session file, rewritten after every run so it
    never shows an old question once the run has moved on.
    """
    store = runtime.store
    # Live output only for a person at a terminal; piped output gets the finished answer only.
    live = runtime.stream and not quiet and sys.stdout.isatty()
    while True:
        if store is not None:
            record = await store.get_chat(chat_id)
            before = record.message_count if record else 0
            execution = runtime.start(agent, question, chat_id=chat_id, owner_id=owner_id, answers=answers or ())
        else:
            before = len(state.messages) if state else 0
            execution = runtime.start(agent, question, state=state, answers=answers or ())
        start = before + (1 if question else 0)  # skip the user's own message
        streamed = False
        try:
            if live:
                streamed = await print_live(execution)
            result = await execution.wait()
        except KeyboardInterrupt:
            logger.warning("exec=%s cancelling on Ctrl-C", execution.execution_id[:8])
            execution.cancel("Ctrl-C")
            result = await execution.wait()
        if not quiet and not live:
            print_transcript(result, start)
        if result.error:
            print(f"error [{result.error.code}]: {result.error.message}", file=sys.stderr)
        if result.persistence_error:
            print(f"warning: the chat could not be saved [{result.persistence_error.code}]: "
                  f"{result.persistence_error.message}", file=sys.stderr)
        if result.output and not streamed:
            print(f"\n{result.output}\n")
        elif streamed:
            print()
        if not quiet:
            print_summary(result)
        if store is None and (session_path is not None or result.awaiting_input):
            session_path = save_session(result.state, sessions_dir, session_path)
        if not result.awaiting_input:
            return result

        # Suspended: nothing is kept in memory. Collect answers and resume from what was saved --
        # exactly what a server does across requests (or a restart).
        answers = collect_answers(result.pending_interactions)
        if answers is None:
            where = chat_id if store is not None else session_path
            print(f"\n(saved; resume any time with: python main.py --resume {where})\n")
            return result
        question = None
        state = None if store is not None else load_session(session_path)


def save_session(state: AgentState, sessions_dir: Path, path: Path | None = None) -> Path:
    path = path or sessions_dir / f"{state.execution_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_session(path: Path) -> AgentState:
    return AgentState.model_validate_json(Path(path).read_text(encoding="utf-8"))


def collect_answers(interactions: list[Interaction]) -> list[Answer] | None:
    """Ask the person in the terminal. Returns None if they choose to answer later."""
    print(f"\n⏸  The agent needs your input ({len(interactions)} item(s)). Type 'later' to stop and resume later.")
    answers = []
    for n, it in enumerate(interactions, 1):
        print(f"\n[{n}/{len(interactions)}] ", end="")
        if it.kind == "approval":
            print(f"APPROVAL  {it.tool} {json.dumps(it.arguments)}\n    {it.message}")
            hint = "a=approve, e=edit, r=reject" if it.allow_edit else "a=approve, r=reject"
            while True:
                choice = _prompt(f"    {hint}> ").lower()
                if choice == "later":
                    return None
                if choice in ("a", "approve"):
                    answers.append(approve(it.id, note=_prompt("    note (optional)> ") or None))
                elif choice in ("e", "edit") and it.allow_edit:
                    raw = _prompt(f"    new arguments as JSON [{json.dumps(it.arguments)}]> ") or json.dumps(it.arguments)
                    try:
                        arguments = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(f"    not valid JSON: {exc.msg}")
                        continue
                    answers.append(edit(it.id, arguments, note=_prompt("    note (optional)> ") or None))
                elif choice in ("r", "reject"):
                    answers.append(reject(it.id, note=_prompt("    reason / note (optional)> ") or None))
                else:
                    continue
                break
        else:
            print(f"QUESTION  {it.message}")
            for i, o in enumerate(it.options, 1):
                print(f"    {i}. {o.label}" + (f" -- {o.description}" if o.description else ""))
            hint = "number" + ("s, comma-separated" if it.multi_select else "") if it.options else ""
            if it.allow_free_text or not it.options:
                hint = f"{hint} or your own answer" if hint else "your answer"
            while True:
                raw = _prompt(f"    {hint} (r=decline)> ")
                if raw.lower() == "later":
                    return None
                if raw.lower() in ("r", "reject"):
                    answers.append(reject(it.id, note=_prompt("    note (optional)> ") or None))
                    break
                value = _pick(raw, it)
                if value is None:
                    print("    please pick one of the numbers" + (" or type an answer" if it.allow_free_text else ""))
                    continue
                answers.append(answer(it.id, value, note=_prompt("    note (optional)> ") or None))
                break
    return answers


def _pick(raw: str, it: Interaction):
    labels = [o.label for o in it.options]
    parts = [p.strip() for p in raw.split(",")] if it.multi_select else [raw.strip()]
    if labels and all(p.isdigit() and 1 <= int(p) <= len(labels) for p in parts):
        picked = [labels[int(p) - 1] for p in parts]
        return picked if it.multi_select else picked[0]
    if raw.strip() and (it.allow_free_text or not labels):
        return raw.strip()
    return None


def _prompt(text: str) -> str:
    try:
        return input(text).strip()
    except EOFError:
        return "later"


async def chat(
    runtime: Runtime, agent: Agent, model: str, quiet: bool, sessions_dir: Path, *,
    chat_id: str | None = None, owner_id: str | None = None,
) -> None:
    store = runtime.store
    if store is not None:
        chat_id = chat_id or new_id()
        saved = await store.load_state(chat_id)
        if saved is not None:
            record = await store.get_chat(chat_id)
            print(f"chat {chat_id}: {record.title or ''} ({record.status.value})")
            print_history(saved.messages)
            print()
    print(f"agent-harness chat — model {model}. Type 'exit' or Ctrl-D to quit, '/reset' to start over.\n")
    state: AgentState | None = None
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in ("exit", "quit"):
            break
        if question == "/reset":
            state = None
            if store is not None:
                chat_id = new_id()
            print("(new conversation)\n")
            continue
        try:
            result = await run_turn(runtime, agent, question, state, quiet, sessions_dir, chat_id=chat_id,
                                    owner_id=owner_id)
        except HarnessError as exc:
            print(f"error [{exc.code}]: {exc.message}\n", file=sys.stderr)
            continue
        # Without a database, keep history across turns unless the run left unanswered tool calls
        # behind (a run left waiting for input is saved to a file; resume it with --resume).
        state = None if result.state.pending_tool_calls() else result.state
    if store is not None and await store.get_chat(chat_id) is not None:
        print(f"(chat saved; reopen it with: python main.py --chat {chat_id})")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent harness against a real model (configured via .env).")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive chat")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the answer (logs: warnings only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (overrides HARNESS_LOG_LEVEL)")
    parser.add_argument("--resume", metavar="CHAT_OR_FILE",
                        help="answer a run that is waiting for your input: a chat id (with a database) or a session file")
    parser.add_argument("--chat", metavar="ID", help="continue (or create) this chat (needs HARNESS_DATABASE_URL)")
    parser.add_argument("--owner", metavar="KEY",
                        help="owner of new chats and filter for --list: a user id, an incident number, ... "
                             "(default HARNESS_OWNER_ID)")
    parser.add_argument("--list", action="store_true", help="list saved chats (needs HARNESS_DATABASE_URL)")
    parser.add_argument("--show", metavar="ID", help="print a saved chat (needs HARNESS_DATABASE_URL)")
    parser.add_argument("--no-stream", action="store_true",
                        help="print the answer once it is complete instead of as it is written (HARNESS_STREAM=false)")
    args = parser.parse_args()

    try:
        load_env_file(ROOT / ".env")
        level = "DEBUG" if args.verbose else "WARNING" if args.quiet else (_env("HARNESS_LOG_LEVEL") or "INFO")
        setup_logging(level, _env("HARNESS_LOG_FILE"))
        config = load_config()
        if args.no_stream:
            config["stream"] = False
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logger.info(
        "config loaded: model=%s provider=%s max_steps=%d timeout=%.0fs settings=%s "
        "max_retries=%d max_execution_time=%s repeat_call=%s/%d max_context_tokens=%s "
        "large_result_handling=%s (per_call=%d per_step=%d) compaction=%s (trigger=%.0f%% keep_recent=%.0f%% model=%s) "
        "parallel_tools=%s (max %d at once) events_file=%s print_events=%s ask_user=%s approval_tools=%s",
        config["model"], "local" if config["local"] else "litellm", config["max_steps"], config["timeout"],
        config["settings"], config["max_retries"], config["max_execution_time"] or "none",
        config["repeat_call_action"], config["repeat_call_threshold"], config["max_context_tokens"] or "none",
        config["large_result_handling"], config["large_result_per_call_tokens"], config["large_result_per_step_tokens"],
        config["compaction"], config["compaction_trigger_ratio"] * 100, config["compaction_keep_recent_ratio"] * 100,
        config["compaction_model"] or "chat model",
        config["parallel_tools"], config["max_parallel_tools"], config["events_file"] or "none",
        config["print_events"], config["ask_user"], ",".join(sorted(config["approval_tools"])) or "none",
    )
    logger.info("streaming: %s (idle timeout %.0fs)", config["stream"], config["stream_idle_timeout"])
    logger.info("persistence: %s (schema=%s lease=%.0fs owner=%s)",
                "postgres" if config["database_url"] else f"session files in {config['sessions_dir']}",
                config["db_schema"], config["lease_seconds"], args.owner or config["owner_id"] or "none")
    if config["compaction"] and not config["max_context_tokens"]:
        logger.warning("HARNESS_COMPACTION is on but HARNESS_MAX_CONTEXT_TOKENS is blank: compaction will only "
                       "run if the provider rejects a request as too long")
    runtime, agent = build(config)
    try:
        return await run_cli(args, runtime, agent, config)
    except HarnessError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2
    finally:
        if runtime.store is not None:
            await runtime.store.close()


async def run_cli(args: argparse.Namespace, runtime: Runtime, agent: Agent, config: dict) -> int:
    store = runtime.store
    sessions_dir = Path(config["sessions_dir"])
    owner = args.owner or config["owner_id"]
    if store is None and (args.chat or args.list or args.show):
        print("--chat, --list and --show need a database: set HARNESS_DATABASE_URL in .env", file=sys.stderr)
        return 2

    if args.list:
        page = await store.list_chats(owner_id=owner, limit=50)
        if not page.items:
            print("no chats" + (f" for owner {owner}" if owner else ""))
        for c in page.items:
            status = "interrupted" if c.interrupted else "running" if c.running else c.status.value
            print(f"{c.id}  {status:<14} {c.updated_at:%Y-%m-%d %H:%M}  {c.message_count:>3} msgs  "
                  f"{c.owner_id or '-':<12} {c.title or ''}")
        return 0

    if args.show:
        record, state = await store.get_chat(args.show), await store.load_state(args.show)
        if record is None or state is None:
            print(f"no chat {args.show}", file=sys.stderr)
            return 2
        print(f"chat {record.id}: {record.title or ''}\n  owner={record.owner_id or '-'} status={record.status.value} "
              f"messages={record.message_count} updated={record.updated_at:%Y-%m-%d %H:%M}")
        print_history(state.messages)
        if record.status is RunStatus.AWAITING_INPUT:
            print(f"\n⏸  waiting for your input; answer with: python main.py --resume {record.id}")
        return 0

    if args.resume:
        # A chat id with a database, a session file without one.
        state = await store.load_state(args.resume) if store is not None else load_session(Path(args.resume))
        if state is None:
            print(f"no chat {args.resume}", file=sys.stderr)
            return 2
        if state.status is not RunStatus.AWAITING_INPUT:
            print(f"{args.resume}: the run is '{state.status.value}', not waiting for input")
            if state.result:
                print(f"\nits last answer:\n\n{state.result}\n")
            return 0 if state.status is RunStatus.COMPLETED else 2
        answers = collect_answers(pending_interactions_of(state))
        if answers is None:
            return 0
        if store is not None:
            result = await run_turn(runtime, agent, None, None, args.quiet, sessions_dir, answers, chat_id=args.resume)
        else:
            result = await run_turn(runtime, agent, None, state, args.quiet, sessions_dir, answers,
                                    session_path=Path(args.resume))
        return 0 if result.ok or result.awaiting_input else 1

    chat_id = args.chat or (new_id() if store is not None else None)
    if args.question:
        result = await run_turn(runtime, agent, " ".join(args.question), None, args.quiet, sessions_dir,
                                chat_id=chat_id, owner_id=owner)
        if store is not None and not args.quiet:
            print(f"  (chat {chat_id}; continue it with: python main.py --chat {chat_id} \"...\")")
        return 0 if result.ok or result.awaiting_input else 1
    await chat(runtime, agent, config["model"], args.quiet, sessions_dir, chat_id=chat_id, owner_id=owner)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
