"""Tool registry: registration, lookup, schema export and tool lifecycle."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator

from agent_harness.errors import ErrorCode, ValidationError
from agent_harness.llm import ToolSchema
from agent_harness.tools import Tool

logger = logging.getLogger(__name__)

# The strictest common denominator across major providers.
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._set_up: list[Tool] = []
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise ValidationError(f"expected a Tool, got {type(tool).__name__}", code=ErrorCode.INVALID_CONFIG)
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not TOOL_NAME_PATTERN.match(name):
            raise ValidationError(
                f"invalid tool name {name!r}: must match {TOOL_NAME_PATTERN.pattern}", code=ErrorCode.INVALID_CONFIG
            )
        if name in self._tools:
            raise ValidationError(f"a tool named '{name}' is already registered", code=ErrorCode.INVALID_CONFIG)
        self._tools[name] = tool
        logger.debug("registered tool %s", name)
        return tool

    def unregister(self, name: str) -> Tool:
        return self._tools.pop(name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(list(self._tools.values()))

    async def setup(self) -> None:
        """Set up every tool in order. On failure, tear down the ones already set up and re-raise."""
        for t in self._tools.values():
            try:
                await t.setup()
            except BaseException as exc:
                logger.error("setup of tool %s failed: %s; tearing down %d tool(s)", t.name, exc, len(self._set_up))
                await self.teardown()
                raise
            self._set_up.append(t)
            logger.debug("tool %s set up", t.name)

    async def teardown(self) -> list[BaseException]:
        """Tear down set-up tools in reverse order. Never raises; returns the errors it contained."""
        errors: list[BaseException] = []
        while self._set_up:
            t = self._set_up.pop()
            try:
                await t.teardown()
                logger.debug("tool %s torn down", t.name)
            except Exception as exc:
                logger.warning("teardown of tool %r failed: %s", t.name, exc)
                errors.append(exc)
        return errors
