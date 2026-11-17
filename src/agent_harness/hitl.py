"""Human-in-the-loop (Phase 12b): suspend a run for a person's input, resume it with the answer.

Nothing waits in memory. When input is needed the run ends with status
`AWAITING_INPUT`; everything needed to continue (the conversation, the open
questions, results already computed in the same batch) lives in
`AgentState.extensions["hitl"]`, so the app saves `state.model_dump_json()`
anywhere and, minutes or a restart later, loads it and calls
`runtime.resume(agent, state, answers=[...])`.

Two ways a run asks for input:

- **Approval** -- a `before_tool` hook returns `ask("why")`. The person can
  approve, edit the arguments, or reject (each with an optional note).
- **Input** -- a tool raises `InputRequired(question, options=...)`, e.g. the
  built-in `ask_user` tool the model calls to ask the person something. On
  resume the tool runs again with `ctx.answer` set.

Calls in the same batch that need no input still run before the run suspends
(their results are held here and committed with the rest, in model order).
A call that is *not* concurrency-safe and comes after one awaiting input is
deferred instead, so it never runs ahead of something it may depend on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from agent_harness.errors import ErrorCode, HarnessError, ValidationError
from agent_harness.messages import new_id, utcnow
from agent_harness.tools import Tool, ToolContext, ToolResult

EXTENSION_KEY = "hitl"

InteractionKind = Literal["approval", "input"]
AnswerAction = Literal["approve", "edit", "reject", "answer"]

# Which answers make sense for which kind of request.
_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    "approval": frozenset({"approve", "edit", "reject"}),
    "input": frozenset({"answer", "reject"}),
}


class InteractionOption(BaseModel):
    label: str
    description: str | None = None


class Interaction(BaseModel):
    """One open request for a person's input, tied to one tool call. JSON-serializable."""

    id: str = Field(default_factory=lambda: new_id("int_"))
    kind: InteractionKind
    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Approval: why it needs approval. Input: the question.
    message: str
    options: list[InteractionOption] = Field(default_factory=list)
    multi_select: bool = False
    # Input: may the person type their own answer instead of picking an option.
    allow_free_text: bool = True
    # Approval: may the person change the arguments before approving.
    allow_edit: bool = True
    step: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """A person's response to one `Interaction`. Build with approve/edit/reject/answer below."""

    interaction_id: str
    action: AnswerAction
    # edit: the replacement arguments (validated against the tool's schema before running).
    arguments: dict[str, Any] | None = None
    # answer: the picked option label(s) or the person's own text.
    value: Any = None
    # Any response may carry a note; the model sees it.
    note: str | None = None
    answered_by: str | None = None
    answered_at: datetime = Field(default_factory=utcnow)


def approve(interaction_id: str, *, note: str | None = None, answered_by: str | None = None) -> Answer:
    return Answer(interaction_id=interaction_id, action="approve", note=note, answered_by=answered_by)


def edit(interaction_id: str, arguments: dict[str, Any], *, note: str | None = None,
         answered_by: str | None = None) -> Answer:
    return Answer(interaction_id=interaction_id, action="edit", arguments=arguments, note=note,
                  answered_by=answered_by)


def reject(interaction_id: str, *, note: str | None = None, answered_by: str | None = None) -> Answer:
    return Answer(interaction_id=interaction_id, action="reject", note=note, answered_by=answered_by)


def answer(interaction_id: str, value: Any, *, note: str | None = None, answered_by: str | None = None) -> Answer:
    return Answer(interaction_id=interaction_id, action="answer", value=value, note=note, answered_by=answered_by)


class InputRequired(HarnessError):
    """Raised by a tool body to ask the person something. The run suspends; on resume the
    tool runs again from the start with `ctx.answer` set (so ask before doing side effects)."""

    def __init__(
        self,
        message: str,
        *,
        options: Iterable[InteractionOption | str | dict[str, Any]] = (),
        multi_select: bool = False,
        allow_free_text: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=ErrorCode.AWAITING_INPUT)
        self.options = [_option(o) for o in options]
        self.multi_select = multi_select
        self.allow_free_text = allow_free_text
        self.metadata = dict(metadata or {})


def _option(value: InteractionOption | str | dict[str, Any]) -> InteractionOption:
    if isinstance(value, InteractionOption):
        return value
    if isinstance(value, str):
        return InteractionOption(label=value)
    return InteractionOption.model_validate(value)


# --- results the executor hands back for calls that did not finish ----------------
# Both use error_code AWAITING_INPUT so the ToolExecutor contract (one ToolResult per call)
# is unchanged; the loop recognizes them and keeps them out of the transcript.


def awaiting_result(interaction: Interaction) -> ToolResult:
    result = ToolResult.error(interaction.call_id, interaction.tool, interaction.message, ErrorCode.AWAITING_INPUT)
    result.metadata["interaction"] = interaction.model_dump(mode="json")
    return result


def deferred_result(call_id: str, name: str) -> ToolResult:
    result = ToolResult.error(call_id, name, "Deferred until pending input is answered.", ErrorCode.AWAITING_INPUT)
    result.metadata["deferred"] = True
    return result


def is_awaiting(result: ToolResult) -> bool:
    return result.error_code == ErrorCode.AWAITING_INPUT


def rejected_result(call_id: str, name: str, kind: str, note: str | None) -> ToolResult:
    what = "this tool call" if kind == "approval" else "this request for input"
    text = f"The user rejected {what}; it was not carried out."
    if note:
        text += f" User's note: {note}"
    return ToolResult.error(call_id, name, text, ErrorCode.TOOL_REJECTED)


def with_note(result: ToolResult, answer_: Answer) -> ToolResult:
    """Tell the model a person was involved (and what they said) alongside the tool's own result."""
    parts = []
    if answer_.action == "edit":
        parts.append(f"The user edited the arguments before approving: {answer_.arguments}")
    elif answer_.action == "approve":
        parts.append("The user approved this call.")
    if answer_.note:
        parts.append(f"User's note: {answer_.note}")
    if not parts:
        return result
    return result.model_copy(update={
        "content": f"{result.content}\n\n[{' '.join(parts)}]",
        "metadata": {**result.metadata, "hitl": {"action": answer_.action, "note": answer_.note}},
    })


# --- what is stored in state.extensions["hitl"] -----------------------------------
#   pending:  [Interaction]               open requests, in model order
#   answers:  {interaction_id: Answer}     submitted, not yet applied
#   held:     {call_id: ToolResult}        finished calls from the suspended batch
#   history:  [{interaction, answer}]      resolved requests (audit trail)


def _data(state: Any) -> dict[str, Any]:
    return state.extensions.setdefault(EXTENSION_KEY, {})


def pending_interactions(state: Any) -> list[Interaction]:
    """Open requests on a suspended run (what the UI should show)."""
    return [Interaction.model_validate(i) for i in (state.extensions.get(EXTENSION_KEY) or {}).get("pending", [])]


def add_pending(state: Any, interaction: Interaction) -> bool:
    """Record a request. Returns False if that call already has one open (asked again on resume)."""
    pending = _data(state).setdefault("pending", [])
    if any(i["call_id"] == interaction.call_id for i in pending):
        return False
    pending.append(interaction.model_dump(mode="json"))
    return True


def submit_answers(state: Any, answers: Iterable[Answer]) -> None:
    """Validate answers against the open requests and store them in state (applied on resume)."""
    open_ = {i.id: i for i in pending_interactions(state)}
    stored = _data(state).setdefault("answers", {})
    for a in answers:
        interaction = open_.get(a.interaction_id)
        if interaction is None:
            raise ValidationError(f"No pending interaction '{a.interaction_id}' on this run.",
                                  code=ErrorCode.INVALID_CONFIG)
        allowed = _ALLOWED_ACTIONS[interaction.kind]
        if a.action not in allowed:
            raise ValidationError(
                f"'{a.action}' is not a valid answer to a {interaction.kind} request (use {', '.join(sorted(allowed))}).",
                code=ErrorCode.INVALID_CONFIG,
            )
        if a.action == "edit":
            if not interaction.allow_edit:
                raise ValidationError(f"Interaction '{a.interaction_id}' does not allow editing.",
                                      code=ErrorCode.INVALID_CONFIG)
            if not isinstance(a.arguments, dict):
                raise ValidationError("An 'edit' answer needs the new arguments as an object.",
                                      code=ErrorCode.INVALID_CONFIG)
        if a.action == "answer":
            if a.value is None and not a.note:
                raise ValidationError("An 'answer' needs a value (an option or the person's own text).",
                                      code=ErrorCode.INVALID_CONFIG)
            labels = {o.label for o in interaction.options}
            picked = a.value if isinstance(a.value, list) else [a.value]
            if isinstance(a.value, list) and not interaction.multi_select:
                raise ValidationError("This question takes a single answer.", code=ErrorCode.INVALID_CONFIG)
            if labels and not interaction.allow_free_text and any(v not in labels for v in picked):
                raise ValidationError(f"Answer must be one of: {', '.join(sorted(labels))}.",
                                      code=ErrorCode.INVALID_CONFIG)
        stored[a.interaction_id] = a.model_dump(mode="json")


def take_answers(state: Any) -> dict[str, tuple[Interaction, Answer]]:
    """Move answered requests from pending to history. Returns {call_id: (interaction, answer)}."""
    data = state.extensions.get(EXTENSION_KEY)
    if not data or not data.get("answers"):
        return {}
    answers = data["answers"]
    still_pending, taken = [], {}
    for raw in data.get("pending", []):
        if raw["id"] in answers:
            interaction, a = Interaction.model_validate(raw), Answer.model_validate(answers.pop(raw["id"]))
            taken[interaction.call_id] = (interaction, a)
            data.setdefault("history", []).append({"interaction": raw, "answer": a.model_dump(mode="json")})
        else:
            still_pending.append(raw)
    data["pending"] = still_pending
    return taken


def hold_results(state: Any, results: Iterable[ToolResult]) -> None:
    held = _data(state).setdefault("held", {})
    for r in results:
        held[r.call_id] = r.model_dump(mode="json")


def get_held(state: Any) -> dict[str, ToolResult]:
    """Finished results from the suspended batch. Left in state until the batch is committed,
    so a run cancelled mid-resume still has them for transcript repair."""
    held = (state.extensions.get(EXTENSION_KEY) or {}).get("held") or {}
    return {cid: ToolResult.model_validate(r) for cid, r in held.items()}


def take_held(state: Any) -> dict[str, ToolResult]:
    held = get_held(state)
    (state.extensions.get(EXTENSION_KEY) or {}).pop("held", None)
    return held


# --- the built-in ask_user tool -----------------------------------------------------


def make_ask_user_tool(name: str = "ask_user") -> Tool:
    """A tool the model calls to ask the person a question, optionally with options to pick from.

    The run suspends until the person answers; the answer (and any note) becomes the
    tool's result. Add the returned tool to `Agent.tools`.
    """

    class AskUser(Tool):
        description = (
            "Ask the user a question when you need a decision, a preference, or information only "
            "they have. Optionally offer options to choose from. The run pauses until they answer; "
            "their answer (and any note) is returned as this tool's result. Ask one clear question "
            "per call, and don't use it for things you can find out with other tools."
        )
        input_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question": {"type": "string", "description": "The question to ask."},
                "options": {
                    "type": "array",
                    "description": "Choices to offer. Omit for an open question.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                },
                "multi_select": {"type": "boolean", "description": "Allow picking several options."},
                "allow_free_text": {
                    "type": "boolean",
                    "description": "Allow an answer that isn't one of the options (default true).",
                },
            },
            "required": ["question"],
        }

        async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
            if ctx.answer is None:
                raise InputRequired(
                    arguments["question"],
                    options=arguments.get("options") or [],
                    multi_select=bool(arguments.get("multi_select", False)),
                    allow_free_text=bool(arguments.get("allow_free_text", True)),
                )
            return {"answer": ctx.answer.value, "note": ctx.answer.note}

    AskUser.name = name
    return AskUser()
