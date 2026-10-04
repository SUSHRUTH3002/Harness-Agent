"""Tool contract and a function-based implementation.

The executor always calls `validate_arguments()` first and passes its output
to `execute()`. Tools may raise; the executor turns every failure into an
error `ToolResult`, so exceptions never reach the agent loop.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic import ValidationError as PydanticValidationError

from agent_harness.cancellation import CancellationToken
from agent_harness.errors import ErrorCode, ValidationError
from agent_harness.llm import ToolSchema

logger = logging.getLogger(__name__)


class ToolAnnotations(BaseModel):
    """Behavioural hints used by later phases (policy, concurrency). `None` means unknown."""

    model_config = ConfigDict(frozen=True)

    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None


class ToolContext(BaseModel):
    """What a tool may see about the execution it runs in. Never the runtime itself."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_id: str
    agent_id: str
    call_id: str
    tool_name: str
    step: int
    # Cooperative cancellation signal (see cancellation.py). 
    cancel: CancellationToken = Field(default_factory=CancellationToken)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    content: str
    is_error: bool = False
    # JSON-compatible structured value; kept in state, not sent to the model.
    data: Any = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def error(cls, call_id: str, name: str, message: str, code: str = ErrorCode.TOOL_ERROR) -> ToolResult:
        return cls(call_id=call_id, name=name, content=message, is_error=True, error_code=str(code))


class Tool(ABC):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    # Declared now so later phases need no interface change.
    concurrency_safe: bool = False
    timeout: float | None = None
    annotations: ToolAnnotations = ToolAnnotations()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return the arguments to pass to `execute`, or raise `ValidationError`."""
        return arguments

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> Any: ...

    async def setup(self) -> None:
        """Acquire resources. Called by the runtime once per run, before the loop."""

    async def teardown(self) -> None:
        """Release resources. Called by the runtime after the run, even on failure."""

    def to_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description, input_schema=copy.deepcopy(self.input_schema))


def _is_tool_context(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation is ToolContext


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) or "Tool"


def _input_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _format_validation_error(tool_name: str, exc: PydanticValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
    )
    return f"Invalid arguments for tool '{tool_name}': {problems}"


class FunctionTool(Tool):
    """Wraps a sync or async function as a tool.

    Arguments are described either by the function's type hints or by an
    explicit Pydantic `args_model`, in which case the function takes a single
    parameter receiving the validated model. A parameter annotated with
    `ToolContext` receives the context and is hidden from the schema.
    Sync functions run in a worker thread so they cannot block the event loop.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
        concurrency_safe: bool = False,
        timeout: float | None = None,
        annotations: ToolAnnotations | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError("FunctionTool requires a callable")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "")
        self.description = description if description is not None else (inspect.getdoc(fn) or "")
        self.concurrency_safe = concurrency_safe
        self.timeout = timeout
        self.annotations = annotations or ToolAnnotations()
        self._is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )

        try:
            hints = get_type_hints(fn, include_extras=True)
        except (NameError, TypeError):
            hints = {}
        self._ctx_param: str | None = None
        params: list[tuple[inspect.Parameter, Any]] = []
        for param in inspect.signature(fn).parameters.values():
            annotation = hints.get(param.name, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            if _is_tool_context(annotation):
                self._ctx_param = param.name
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(f"tool '{self.name}': *args/**kwargs parameters are not supported")
            params.append((param, annotation))

        self._model_param: str | None = None
        if args_model is not None:
            if len(params) != 1:
                raise TypeError(f"tool '{self.name}': a function using args_model must take exactly one argument")
            self._model = args_model
            self._model_param = params[0][0].name
        else:
            fields: dict[str, Any] = {
                p.name: (annotation, ... if p.default is inspect.Parameter.empty else p.default)
                for p, annotation in params
            }
            self._model = create_model(
                f"{_camel(self.name)}Arguments", __config__=ConfigDict(extra="forbid"), **fields
            )
        self.input_schema = _input_schema(self._model)

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            model = self._model.model_validate(arguments)
        except PydanticValidationError as exc:
            raise ValidationError(
                _format_validation_error(self.name, exc),
                code=ErrorCode.INVALID_ARGS,
                details={"errors": exc.errors(include_url=False, include_context=False, include_input=False)},
            ) from exc
        if self._model_param is not None:
            return {self._model_param: model}
        return {field: getattr(model, field) for field in type(model).model_fields}

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> Any:
        kwargs = dict(arguments)
        if self._ctx_param is not None:
            kwargs[self._ctx_param] = ctx
        if self._is_async:
            return await self.fn(**kwargs)
        logger.info("tool %s: running sync function in a worker thread", self.name)
        result = await asyncio.to_thread(self.fn, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_model: type[BaseModel] | None = None,
    concurrency_safe: bool = False,
    timeout: float | None = None,
    annotations: ToolAnnotations | None = None,
) -> Any:
    """Decorator turning a function into a `FunctionTool`. Usable as `@tool` or `@tool(...)`."""

    def wrap(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            f,
            name=name,
            description=description,
            args_model=args_model,
            concurrency_safe=concurrency_safe,
            timeout=timeout,
            annotations=annotations,
        )

    return wrap(fn) if fn is not None else wrap


def normalize_output(call_id: str, name: str, value: Any) -> ToolResult:
    """Convert whatever a tool returned into a `ToolResult` with JSON-compatible `data`."""
    if isinstance(value, ToolResult):
        return value.model_copy(update={"call_id": call_id, "name": name})
    if isinstance(value, str):
        return ToolResult(call_id=call_id, name=name, content=value)
    if value is None:
        return ToolResult(call_id=call_id, name=name, content="")
    if isinstance(value, BaseModel):
        data = value.model_dump(mode="json")
        return ToolResult(call_id=call_id, name=name, content=json.dumps(data, ensure_ascii=False), data=data)
    content = json.dumps(value, ensure_ascii=False, default=str)
    return ToolResult(call_id=call_id, name=name, content=content, data=json.loads(content))
