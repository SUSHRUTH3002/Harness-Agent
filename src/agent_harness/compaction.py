"""Compaction: keep long sessions within the context budget without forgetting them.

When the conversation reaches `agent.limits.compaction_trigger_ratio` of the
budget, the history is split into three parts:

    [first message] [ REGION (older middle) ] [ recent tail ]
       always kept     summarized + archived    kept verbatim

The region's original messages are saved to the artifact store (an archive
the model can search with `read_artifact`), and one extra LLM call writes an
evidence-preserving summary of it. From then on, the model is sent:
first message + summary + recent tail. A later compaction folds the previous
summary into a new one (a rolling summary) and archives the newly covered
messages.

`state.messages` is never edited (design.md D3): the compaction checkpoint
lives in `state.extensions["compaction"]` and `compacted_view()` applies it
when the context is built. If summarizing fails, nothing changes and Phase 3
dropping (context.py `_fit_budget`) still keeps the request in budget.

Nothing here runs unless a `Compactor` is attached to a `Runtime`
(see runtime.py `compactor=`).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence

from agent_harness.agent import Agent
from agent_harness.context import CharTokenCounter, TokenCounter, _message_tokens, _segment_history
from agent_harness.llm import FinishReason, LLMProvider, LLMRequest, LLMResponse, Usage
from agent_harness.messages import Message, Role, short_id, utcnow
from agent_harness.results import ArtifactStore
from agent_harness.state import AgentState

logger = logging.getLogger(__name__)

EXTENSION_KEY = "compaction"

SUMMARY_SYSTEM_PROMPT = (
    "You compact the working memory of an AI agent in the middle of a long task, such as "
    "debugging or an incident root-cause analysis that pulls data from many sources. The agent "
    "will continue the task with only your summary in place of the messages you are given, so "
    "anything you leave out is forgotten. Be brief in wording, never in facts."
)

SUMMARY_INSTRUCTIONS = """\
Write a summary of the conversation below that lets the agent continue the task without it.

Use exactly these sections, leaving a section out only if it would be empty:

## User requests
Every request the user made, quoted verbatim, oldest first.

## Timeline
Events in time order, each with its exact timestamp as it appeared in the data.

## Evidence
Concrete findings: exact error messages, ids, hostnames, versions, metric values, counts, file paths,
commands. Copy identifiers, numbers and error text exactly -- never round, shorten or paraphrase them.
Name the tool/source each finding came from.

## Sources checked
Each tool or source queried, its key arguments, and what it returned -- including searches that found
nothing and calls that failed.

## Hypotheses
Current hypotheses with the evidence for and against each. Mark ruled-out ones as ruled out, and why.

## Conclusions so far

## Open questions and next steps

Rules:
- If a previous summary is given, carry everything in it forward unless the new messages supersede it.
- Do not invent anything. If something is uncertain, say so.
- A result that was too large and saved as an artifact is referenced by its artifact id; keep that id.
- Output only the summary.
"""

# Per-message cap on what the summarizer is sent (not on the archive, which keeps everything).
# Keeps the summary request itself from overflowing when one message is enormous.
_SUMMARIZER_MESSAGE_CHARS = 24_000
# After a failed summary, skip this many steps before trying again, instead of paying for
# another failing (and retried) LLM call on every step.
_RETRY_AFTER_FAILURE_STEPS = 3


def compaction_data(state: AgentState) -> dict[str, Any] | None:
    """The compaction checkpoint, or None if this state has never been compacted."""
    data = state.extensions.get(EXTENSION_KEY)
    return data if data and data.get("covered_until") else None


def render_summary_message(data: dict[str, Any]) -> Message:
    """The message that stands in for every compacted region: summary + the archive index."""
    lines = [
        f"[Context compacted {data['count']} time(s). The earlier part of this conversation was replaced by "
        "this summary; the most recent messages follow it unchanged.]",
        "",
        data["summary"].strip(),
    ]
    archives = data.get("archives") or []
    if archives:
        lines += [
            "",
            "## Archives",
            "The original messages behind this summary are saved word for word. When you need an exact "
            "detail the summary doesn't have, search them with the read_artifact tool "
            "(artifact_id + search).",
        ]
        for a in archives:
            first, last = a["messages"]
            tools = ", ".join(a["tools"]) or "none"
            lines.append(f"- {a['artifact_id']}: messages {first}-{last} · tools: {tools} · ~{a['tokens']} tokens")
    return Message.user("\n".join(lines), metadata={"source": "harness", "reason": "compaction_summary"})


def compacted_view(state: AgentState) -> list[Message]:
    """`state.messages` as the model should see it: first message + summary + everything not yet compacted."""
    data = compaction_data(state)
    if data is None or not state.messages:
        return list(state.messages)
    return [state.messages[0], render_summary_message(data), *state.messages[data["covered_until"]:]]


def _format_message(index: int, message: Message, limit: int | None) -> str:
    if message.role is Role.TOOL:
        head = f"[message {index} · tool result: {message.name or '?'}{' (error)' if message.is_error else ''}]"
    else:
        head = f"[message {index} · {message.role.value}]"
    body = message.text
    if limit is not None and len(body) > limit:
        half = limit // 2
        body = (
            f"{body[:half]}\n...({len(body) - limit} characters omitted here; the archive has the full text)...\n"
            f"{body[-half:]}"
        )
    lines = [head]
    if body:
        lines.append(body)
    for call in message.tool_calls:
        args = json.dumps(call.arguments, ensure_ascii=False) if call.arguments is not None else call.raw_arguments
        lines.append(f"→ called {call.name}({args or ''})")
    return "\n".join(lines)


def _transcript(messages: Sequence[Message], first_index: int, limit: int | None = None) -> str:
    return "\n\n".join(_format_message(first_index + i, m, limit) for i, m in enumerate(messages))


def _tool_names(messages: Sequence[Message]) -> list[str]:
    return sorted({c.name for m in messages for c in m.tool_calls})


class Compactor:
    """Summarizes and archives the older middle of a conversation once it grows past the trigger.

    `llm`: the provider used to write summaries. None (default) = the run's own chat provider.
    `model`: model id for the summary request. None = `agent.model` (the chat model).
    `store`: where each compacted region's original messages are archived. Pass the same store
        `make_read_artifact_tool` was built with so the model can search archives; None = no archive.
    `summary_max_tokens`: output-token cap for the summary itself.
    """

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        model: str | None = None,
        store: ArtifactStore | None = None,
        token_counter: TokenCounter | None = None,
        summary_max_tokens: int = 4096,
    ) -> None:
        self.llm = llm
        self.model = model
        self.store = store
        self.token_counter = token_counter or CharTokenCounter()
        self.summary_max_tokens = summary_max_tokens

    # --- measuring ------------------------------------------------------------------------

    def record_usage(self, state: AgentState, request: LLMRequest, history_length: int, usage: Usage) -> None:
        """Remember the provider-reported prompt size of the request just sent.

        The next size check is then that real number plus an estimate for only the messages
        added since, instead of a chars/4 estimate of everything. Skipped when the provider
        reports no usage, or when Phase 3 dropped messages from that request (its real size
        then understates the full history).
        """
        data = state.extensions.setdefault(EXTENSION_KEY, {})
        if usage.input_tokens <= 0 or request.metadata.get("context", {}).get("groups_dropped"):
            data.pop("usage_anchor", None)
            return
        data["usage_anchor"] = {
            "input_tokens": usage.input_tokens,
            "history_length": history_length,
            "covered_until": data.get("covered_until", 0),
        }

    def estimate_tokens(self, state: AgentState, request: LLMRequest) -> tuple[int, str]:
        """Prompt size of the next request *before* any Phase 3 dropping, and how it was measured."""
        counter = self.token_counter
        data = state.extensions.get(EXTENSION_KEY) or {}
        anchor = data.get("usage_anchor")
        if (
            anchor
            and anchor["covered_until"] == data.get("covered_until", 0)
            and anchor["history_length"] <= len(state.messages)
        ):
            added = state.messages[anchor["history_length"]:]
            return anchor["input_tokens"] + sum(_message_tokens(m, counter) for m in added), "measured"

        system = [m for m in request.messages if m.role is Role.SYSTEM]
        tools = sum(
            counter.count(t.name) + counter.count(t.description) + counter.count(json.dumps(t.input_schema))
            for t in request.tools
        )
        view = compacted_view(state)
        return sum(_message_tokens(m, counter) for m in [*system, *view]) + tools, "estimated"

    # --- compacting -------------------------------------------------------------------------

    async def maybe_compact(self, agent: Agent, state: AgentState, request: LLMRequest, llm: LLMProvider) -> bool:
        """Compact if the next request would pass the trigger. Returns True if it compacted."""
        budget = agent.limits.max_context_tokens
        if budget is None:
            return False
        if state.step < (state.extensions.get(EXTENSION_KEY) or {}).get("retry_after_step", 0):
            return False  # a recent attempt failed; Phase 3 dropping covers the steps in between
        usable = budget - agent.limits.reserved_output_tokens
        trigger = int(usable * agent.limits.compaction_trigger_ratio)
        tokens, how = self.estimate_tokens(state, request)
        if tokens < trigger:
            logger.debug("exec=%s step=%d context ~%d tokens (%s) under compaction trigger %d",
                         short_id(state.execution_id), state.step, tokens, how, trigger)
            return False
        logger.info(
            "exec=%s step=%d context ~%d tokens (%s) reached the compaction trigger (%d = %.0f%% of %d); compacting",
            short_id(state.execution_id), state.step, tokens, how, trigger,
            agent.limits.compaction_trigger_ratio * 100, usable,
        )
        return await self.compact(agent, state, llm, target_tokens=usable, current_tokens=tokens, reason="trigger")

    async def compact(
        self,
        agent: Agent,
        state: AgentState,
        llm: LLMProvider,
        *,
        target_tokens: int,
        current_tokens: int,
        reason: str,
    ) -> bool:
        """Summarize + archive the region. Returns False (and changes nothing) if there is nothing
        to compact or the summary could not be written."""
        exec_id = short_id(state.execution_id)
        counter = self.token_counter
        data = state.extensions.setdefault(EXTENSION_KEY, {})
        start = data.get("covered_until") or 1
        tail_start = self._tail_start(state.messages, start, int(target_tokens * agent.limits.compaction_keep_recent_ratio))
        if tail_start is None:
            logger.warning("exec=%s step=%d compaction (%s): nothing to compact -- the recent tail is all there is",
                           exec_id, state.step, reason)
            return False

        region = state.messages[start:tail_start]
        region_tokens = sum(_message_tokens(m, counter) for m in region)
        previous_summary = data.get("summary")
        count = data.get("count", 0) + 1

        try:
            summary = await self._summarize(agent, llm, region, start, previous_summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            data["retry_after_step"] = state.step + _RETRY_AFTER_FAILURE_STEPS
            logger.warning(
                "exec=%s step=%d compaction (%s) failed while summarizing, keeping the history as is "
                "(Phase 3 dropping still applies; next attempt from step %d): %s",
                exec_id, state.step, reason, data["retry_after_step"], exc,
            )
            return False
        if not summary.strip():
            logger.warning("exec=%s step=%d compaction (%s): the model returned an empty summary; skipped",
                           exec_id, state.step, reason)
            return False

        archive = await self._archive(state, region, start, tail_start - 1, region_tokens, count)

        old_summary_tokens = _message_tokens(render_summary_message(data), counter) if previous_summary else 0
        data.update(
            summary=summary,
            covered_until=tail_start,
            count=count,
            archives=[*data.get("archives", []), *([archive] if archive else [])],
        )
        data.pop("usage_anchor", None)
        data.setdefault("history", []).append({
            "at": utcnow().isoformat(), "step": state.step, "reason": reason,
            "messages": [start, tail_start - 1], "region_tokens": region_tokens,
            "summary_tokens": counter.count(summary), "archive": archive["artifact_id"] if archive else None,
        })

        summary_tokens = _message_tokens(render_summary_message(data), counter)
        after = current_tokens - region_tokens - old_summary_tokens + summary_tokens
        logger.warning(
            "exec=%s step=%d compacted messages %d-%d (~%d tokens, tools: %s) into a ~%d-token summary%s; "
            "context ~%d -> ~%d tokens",
            exec_id, state.step, start, tail_start - 1, region_tokens, ", ".join(_tool_names(region)) or "none",
            summary_tokens, f", archived as '{archive['artifact_id']}'" if archive else "", current_tokens, after,
        )
        return True

    def _tail_start(self, messages: Sequence[Message], start: int, keep_tokens: int) -> int | None:
        """Index where the verbatim tail begins: at least `keep_tokens` of the newest messages,
        on a segment boundary (a tool call is never split from its results). None if the region
        would be empty."""
        counter = self.token_counter
        segments = _segment_history(messages[start:])
        if len(segments) < 2:
            return None
        offsets, i = [], start
        for s in segments:
            offsets.append(i)
            i += len(s.messages)
        kept = 0
        cut = len(segments) - 1
        for k in range(len(segments) - 1, 0, -1):
            kept += sum(_message_tokens(m, counter) for m in segments[k].messages)
            cut = k
            if kept >= keep_tokens:
                break
        return offsets[cut]

    async def _summarize(
        self, agent: Agent, llm: LLMProvider, region: Sequence[Message], first_index: int, previous: str | None
    ) -> str:
        parts = [SUMMARY_INSTRUCTIONS]
        if previous:
            parts.append(f"<previous_summary>\n{previous.strip()}\n</previous_summary>")
        parts.append(f"<conversation>\n{_transcript(region, first_index, _SUMMARIZER_MESSAGE_CHARS)}\n</conversation>")
        request = LLMRequest(
            messages=[Message.system(SUMMARY_SYSTEM_PROMPT), Message.user("\n\n".join(parts))],
            model=self.model or agent.model,
            settings={**agent.model_settings, "max_tokens": self.summary_max_tokens},
            metadata={"purpose": "compaction"},
        )
        provider = self.llm or llm
        if agent.limits.llm_timeout is None:
            response: LLMResponse = await provider.generate(request)
        else:
            async with asyncio.timeout(agent.limits.llm_timeout):
                response = await provider.generate(request)
        if response.finish_reason is FinishReason.LENGTH:
            logger.warning("compaction summary hit summary_max_tokens=%d and was cut off; using it as is",
                           self.summary_max_tokens)
        return response.message.text

    async def _archive(
        self, state: AgentState, region: Sequence[Message], first: int, last: int, tokens: int, count: int
    ) -> dict[str, Any] | None:
        if self.store is None:
            return None
        tools = _tool_names(region)
        header = "\n".join([
            f"# Compacted region {count}",
            f"# Execution: {state.execution_id}",
            f"# Archived at: {utcnow().isoformat()} (step {state.step})",
            f"# Contains: messages {first}-{last} of the full history, word for word",
            f"# Tools called: {', '.join(tools) or 'none'}",
            f"# Size: ~{tokens} estimated tokens",
        ])
        try:
            artifact_id = await self.store.save(f"{header}\n\n{_transcript(region, first)}\n",
                                                name_hint=f"compacted_region_{count}")
        except Exception as exc:
            # The summary is still worth having; the model just can't look up the originals.
            logger.warning("exec=%s compaction: archiving region %d failed: %s", short_id(state.execution_id), count, exc)
            return None
        return {"artifact_id": artifact_id, "messages": [first, last], "tools": tools, "tokens": tokens,
                "step": state.step, "created_at": utcnow().isoformat()}
