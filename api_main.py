"""Run the Chat API (Phase 11a): the same agent as main.py, served over HTTP and WebSocket.

    pip install -e ".[api,postgres]"
    python api_main.py                       # http://127.0.0.1:8000  (interactive docs at /docs)
    python api_main.py --host 0.0.0.0 --port 9000

Configured from .env like main.py (model, tools, limits, HARNESS_DATABASE_URL, ...) plus the
HARNESS_API_* settings in .env.example. Without HARNESS_DATABASE_URL chats are kept in memory and
lost when the server stops.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

import main as cli
from agent_harness.api.server import create_app
from agent_harness.stores.memory import InMemoryStore

logger = logging.getLogger("api_main")


def build_app():
    config = cli.load_config()
    runtime, agent = cli.build(config)
    if runtime.store is None:
        logger.warning("HARNESS_DATABASE_URL is not set: chats are kept in memory and lost when the server stops")
        runtime.store = InMemoryStore()
    origins = [o.strip() for o in (cli._env("HARNESS_API_CORS_ORIGINS") or "").split(",") if o.strip()]
    token = cli._env("HARNESS_API_TOKEN")
    if not token:
        logger.warning("HARNESS_API_TOKEN is not set: anyone who can reach this server can use it")
    app = create_app(
        runtime, agent, token=token, cors_origins=origins,
        max_active_runs=cli._number("HARNESS_API_MAX_ACTIVE_RUNS", int, 50) or None,
        keep_finished=cli._number("HARNESS_API_KEEP_FINISHED", float, 60.0),
        docs=cli._bool("HARNESS_API_DOCS", True),
    )
    return app, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the agent over HTTP and WebSocket (configured via .env).")
    parser.add_argument("--host", help="address to listen on (default HARNESS_API_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default HARNESS_API_PORT or 8000)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    try:
        cli.load_env_file(cli.ROOT / ".env")
        cli.setup_logging("DEBUG" if args.verbose else (cli._env("HARNESS_LOG_LEVEL") or "INFO"),
                          cli._env("HARNESS_LOG_FILE"))
        app, runtime = build_app()
    except cli.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    host = args.host or cli._env("HARNESS_API_HOST") or "127.0.0.1"
    port = args.port or cli._number("HARNESS_API_PORT", int, 8000)
    # Our own logging setup stays in charge; uvicorn only adds its access lines.
    uvicorn.run(app, host=host, port=port, log_config=None, access_log=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
