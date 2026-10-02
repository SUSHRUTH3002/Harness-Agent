"""Run the agent harness against a real model, configured from .env.

    python main.py "What's the weather in Tokyo in Fahrenheit?"   # one question
    python main.py                                                # interactive chat
    python main.py -q "..."                                       # answer only, no transcript

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

from agent_harness import Agent, AgentLimits, AgentState, Role, RunResult, Runtime
from agent_harness.providers.litellm_provider import LiteLLMProvider
from agent_harness.providers.local_provider import LocalProvider
from agent_harness.testing import calculator, get_weather, web_search
from agent_harness.tools_dir.agent_tools import fetch_resource_metadata

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
    "instead of guessing. Answer concisely."
)


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
    return {
        "model": model,
        "local": local,
        "api_base": api_base,
        "api_key": api_key,
        "timeout": _number("HARNESS_TIMEOUT", float, 120.0),
        "max_steps": _number("HARNESS_MAX_STEPS", int, 15),
        "settings": settings,
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
    agent = Agent(
        name="assistant",
        instructions=INSTRUCTIONS,
        tools=[calculator, get_weather, web_search, fetch_resource_metadata],
        model_settings=config["settings"],
        limits=AgentLimits(max_steps=config["max_steps"]),
    )
    return Runtime(provider), agent


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


async def ask(runtime: Runtime, agent: Agent, question: str, state: AgentState | None, quiet: bool) -> RunResult:
    start = len(state.messages) + 1 if state else 1  # skip the user's own message
    result = await runtime.run(agent, question, state=state)
    if not quiet:
        print_transcript(result, start)
    if result.error:
        print(f"error [{result.error.code}]: {result.error.message}", file=sys.stderr)
    if result.output:
        print(f"\n{result.output}\n")
    if not quiet:
        print_summary(result)
    return result


async def chat(runtime: Runtime, agent: Agent, model: str, quiet: bool) -> None:
    print(f"agent-harness chat — model {model}. Type 'exit' or Ctrl-D to quit, '/reset' to start over.\n")
    state: AgentState | None = None
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in ("exit", "quit"):
            return
        if question == "/reset":
            state = None
            print("(conversation cleared)\n")
            continue
        result = await ask(runtime, agent, question, state, quiet)
        # Keep history across turns unless the run left unanswered tool calls behind.
        state = None if result.state.pending_tool_calls() else result.state


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent harness against a real model (configured via .env).")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive chat")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the answer (logs: warnings only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (overrides HARNESS_LOG_LEVEL)")
    args = parser.parse_args()

    try:
        load_env_file(ROOT / ".env")
        level = "DEBUG" if args.verbose else "WARNING" if args.quiet else (_env("HARNESS_LOG_LEVEL") or "INFO")
        setup_logging(level, _env("HARNESS_LOG_FILE"))
        config = load_config()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logger.info(
        "config loaded: model=%s provider=%s max_steps=%d timeout=%.0fs settings=%s",
        config["model"], "local" if config["local"] else "litellm", config["max_steps"], config["timeout"],
        config["settings"],
    )
    runtime, agent = build(config)

    if args.question:
        result = await ask(runtime, agent, " ".join(args.question), None, args.quiet)
        return 0 if result.ok else 1
    await chat(runtime, agent, config["model"], args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
