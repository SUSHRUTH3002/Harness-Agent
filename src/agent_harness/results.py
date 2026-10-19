"""Large tool result handling.

A tool result over `agent.limits.large_result_per_call_tokens` -- or one of
several results that together exceed `large_result_per_step_tokens` even
though none is individually over -- gets its full content saved to an
`ArtifactStore` and replaced, in what the model sees, with a short preview
plus a pointer. The `read_artifact` tool lets the model retrieve it later,
deliberately, if it decides it actually needs to.

Nothing here runs unless a `LargeResultProcessor` is attached to a `Runtime`
(see runtime.py `result_processor=`) -- with none attached, behavior is
identical to Phase 1-6.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_harness.agent import Agent
from agent_harness.context import CharTokenCounter, TokenCounter
from agent_harness.errors import ToolError
from agent_harness.messages import ToolCall, short_id
from agent_harness.tools import Tool, ToolAnnotations, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _new_artifact_id(name_hint: str) -> str:
    safe_hint = _SAFE_CHARS.sub("_", name_hint)[:48] or "artifact"
    return f"{safe_hint}_{uuid.uuid4().hex[:8]}"


@runtime_checkable
class ArtifactStore(Protocol):
    async def save(self, content: str, *, name_hint: str) -> str:
        """Save `content` and return an artifact id that `read` can look it up by later."""
        ...

    async def read(self, artifact_id: str) -> str:
        """Return the full content previously saved under `artifact_id`. Raises `ToolError` if missing."""
        ...


class InMemoryArtifactStore:
    """Default store: a plain dict. Lost when the process exits -- fine for a single run or
    an interactive session; use `FileArtifactStore` for anything that needs to survive longer."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def save(self, content: str, *, name_hint: str) -> str:
        artifact_id = _new_artifact_id(name_hint)
        self._data[artifact_id] = content
        return artifact_id

    async def read(self, artifact_id: str) -> str:
        try:
            return self._data[artifact_id]
        except KeyError:
            raise ToolError(f"artifact '{artifact_id}' not found") from None


class FileArtifactStore:
    """Writes each artifact to its own file under `directory`. Survives process restarts."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        # Re-sanitize even though _new_artifact_id already produces safe ids: artifact_id may
        # come back from the model as a tool argument, so it is untrusted at read time.
        safe_id = _SAFE_CHARS.sub("_", artifact_id)
        return self.directory / f"{safe_id}.txt"

    async def save(self, content: str, *, name_hint: str) -> str:
        artifact_id = _new_artifact_id(name_hint)
        await asyncio.to_thread(self._path(artifact_id).write_text, content, "utf-8")
        return artifact_id

    async def read(self, artifact_id: str) -> str:
        path = self._path(artifact_id)
        if not path.exists():
            raise ToolError(f"artifact '{artifact_id}' not found")
        return await asyncio.to_thread(path.read_text, "utf-8")


def _preview(content: str, chars: int) -> str:
    if chars <= 0 or len(content) <= 2 * chars:
        return content
    omitted = len(content) - 2 * chars
    return f"{content[:chars]}\n...({omitted} characters omitted)...\n{content[-chars:]}"


class LargeResultProcessor:
    """Offloads oversized tool results for one step's batch at once.

    Runs once per step, after every call in that step has a result and before
    they're added to history -- it needs to see the whole batch together to
    apply the per-step combined threshold (one call at a time, as a hook
    would see it, can't tell whether *together* several calls are too much).
    """

    def __init__(self, *, store: ArtifactStore | None = None, token_counter: TokenCounter | None = None) -> None:
        self.store = store or InMemoryArtifactStore()
        self.token_counter = token_counter or CharTokenCounter()

    async def process(
        self,
        agent: Agent,
        calls: list[ToolCall],
        results: list[ToolResult],
        *,
        execution_id: str,
        step: int,
        exempt_tools: frozenset[str] = frozenset(),
    ) -> list[ToolResult]:
        """`exempt_tools`: names of tools whose results always stay inline (`Tool.exempt_from_offload`).
        They are never offloaded and don't count toward the per-step total."""
        per_call = agent.limits.large_result_per_call_tokens
        per_step = agent.limits.large_result_per_step_tokens
        if per_call is None and per_step is None:
            return results

        counter = self.token_counter
        sizes = [counter.count(r.content) for r in results]
        # Errors and exempt tools are never candidates: the model must see them directly.
        eligible = {i for i, r in enumerate(results) if not r.is_error and r.name not in exempt_tools}
        offload: set[int] = set()

        if per_call is not None:
            offload = {i for i in eligible if sizes[i] > per_call}

        if per_step is not None:
            remaining = sum(sizes[i] for i in eligible if i not in offload)
            if remaining > per_step:
                # Largest first, among what's left, until it fits.
                order = sorted((i for i in eligible if i not in offload), key=lambda i: sizes[i], reverse=True)
                for i in order:
                    if remaining <= per_step:
                        break
                    remaining -= sizes[i]
                    offload.add(i)

        if not offload:
            return results

        preview_chars = agent.limits.large_result_preview_chars
        exec_id = short_id(execution_id)
        summaries = []
        new_results = list(results)
        for i in sorted(offload):
            new_results[i] = await self._offload(calls[i], results[i], sizes[i], preview_chars)
            summaries.append(f"{results[i].name} (~{sizes[i]} tokens)")
        logger.warning(
            "exec=%s step=%d offloaded %d large tool result(s): %s",
            exec_id, step, len(offload), "; ".join(summaries),
        )
        return new_results

    async def _offload(self, call: ToolCall, result: ToolResult, tokens: int, preview_chars: int) -> ToolResult:
        name_hint = f"{result.name}_{short_id(call.id)}"
        artifact_id = await self.store.save(result.content, name_hint=name_hint)
        preview = _preview(result.content, preview_chars)
        content = (
            f"[Result too large to include in full: ~{tokens} estimated tokens, "
            f"{len(result.content)} characters.]\n"
            f"Preview (first/last {preview_chars} characters):\n{preview}\n\n"
            f"Full result saved as artifact '{artifact_id}'. Use the read_artifact tool "
            f"(artifact_id=\"{artifact_id}\") to read specific parts of it if you need to."
        )
        return result.model_copy(
            update={"content": content, "metadata": {**result.metadata, "artifact_id": artifact_id, "offloaded": True}}
        )


def make_read_artifact_tool(store: ArtifactStore, *, max_chars: int = 4000) -> Tool:
    """Build a `read_artifact` tool bound to `store`. Pass the same store instance you gave
    `LargeResultProcessor`, and add the returned tool to `Agent.tools`, or the model has no
    way to retrieve anything that gets offloaded.

    `max_chars` caps the content returned by one call (a larger `length` is clamped, and
    `has_more` + `next_start` tell the model where to continue). A tail under 10% of
    `max_chars` is folded into the current page rather than left for another call. Its
    results are exempt from offloading, so this cap is the only thing keeping a single read
    from flooding the conversation.
    """
    search_window = max_chars // 2

    class ReadArtifact(Tool):
        name = "read_artifact"
        exempt_from_offload = True
        description = (
            "Read a large tool result that was previously offloaded (see a tool result starting "
            "with 'Result too large...'). Use `search` to jump to a specific part, or `start`/"
            f"`length` to page through it. At most {max_chars} characters are returned per call; "
            "when `has_more` is true, call again with `start` set to the returned `next_start`."
        )
        input_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "artifact_id": {"type": "string", "description": "The id given in the offload message."},
                "search": {
                    "type": "string",
                    "description": "Find this text (case-insensitive) and return a window of content around its first match.",
                },
                "start": {"type": "integer", "default": 0, "description": "Character offset to start reading from."},
                "length": {
                    "type": "integer", "default": max_chars,
                    "description": f"Characters to return (capped at {max_chars}).",
                },
            },
            "required": ["artifact_id"],
        }
        annotations = ToolAnnotations(read_only=True, idempotent=True)

        async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            content = await store.read(arguments["artifact_id"])
            total = len(content)

            search = arguments.get("search")
            if search:
                index = content.lower().find(str(search).lower())
                if index == -1:
                    return {"found": False, "total_characters": total, "message": f"'{search}' was not found."}
                half = search_window // 2
                start = max(0, index - half)
                end = min(total, start + search_window)
                return {
                    "found": True, "match_at": index, "total_characters": total,
                    "content": content[start:end], "range": [start, end],
                }

            start = max(0, int(arguments.get("start", 0)))
            length = min(max_chars, max(1, int(arguments.get("length", max_chars))))
            end = min(total, start + length)
            # A leftover tail under 10% of a page isn't worth another model call; include it now.
            if 0 < total - end <= max_chars // 10:
                end = total
            page: dict[str, Any] = {"total_characters": total, "range": [start, end], "has_more": end < total}
            if end < total:
                # Tell the model exactly where to continue, so it doesn't overlap or leave gaps.
                page["next_start"] = end
                page["remaining_characters"] = total - end
            page["content"] = content[start:end]
            return page

    return ReadArtifact()
