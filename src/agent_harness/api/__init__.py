"""Chat API (Phase 11a): HTTP + WebSocket on top of the harness.

Not re-exported from `agent_harness` itself: the core depends only on pydantic, and the web app
(`agent_harness.api.server.create_app`) needs FastAPI (`pip install -e ".[api]"`). `ChatService`
and the models need no web framework.
"""

from agent_harness.api.broadcast import Gap, RunBroadcast
from agent_harness.api.models import (
    AnswerIn,
    AnswersIn,
    ChatCreate,
    ChatDetail,
    ChatList,
    ChatOut,
    ChatUpdate,
    MessageCreate,
    MessageOut,
    RunStarted,
)
from agent_harness.api.service import BusyError, ChatService, NotFoundError, TooManyRunsError

__all__ = [
    "AnswerIn", "AnswersIn", "BusyError", "ChatCreate", "ChatDetail", "ChatList", "ChatOut", "ChatService",
    "ChatUpdate", "Gap", "MessageCreate", "MessageOut", "NotFoundError", "RunBroadcast", "RunStarted",
    "TooManyRunsError",
]
