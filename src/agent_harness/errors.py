"""Structured error model.

Harness failures are expressed as `HarnessError` subclasses carrying a
provider-neutral `code`. What gets stored in state and results is the
serializable `ErrorInfo`, never a raw exception.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Well-known error codes. Codes are plain strings, so adapters may add their own."""

    UNKNOWN = "UNKNOWN"
    # LLM
    LLM_ERROR = "LLM_ERROR"
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
    INVALID_REQUEST = "INVALID_REQUEST"
    SERVER = "SERVER"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    # Tools
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGS = "INVALID_ARGS"
    TOOL_ERROR = "TOOL_ERROR"
    # Runtime / configuration
    EXECUTION_ERROR = "EXECUTION_ERROR"
    INVALID_CONFIG = "INVALID_CONFIG"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    CANCELLED = "CANCELLED"


class ErrorInfo(BaseModel):
    """Serializable description of a failure, safe to store and return."""

    code: str
    message: str
    type: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException) -> ErrorInfo:
        if isinstance(exc, HarnessError):
            return cls(
                code=exc.code,
                message=exc.message,
                type=type(exc).__name__,
                retryable=exc.retryable,
                details=exc.details,
            )
        return cls(code=ErrorCode.UNKNOWN, message=str(exc) or type(exc).__name__, type=type(exc).__name__)


class HarnessError(Exception):
    default_code: str = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = str(code or self.default_code)
        self.retryable = retryable
        self.details = dict(details or {})

    def to_info(self) -> ErrorInfo:
        return ErrorInfo.from_exception(self)


class LLMError(HarnessError):
    default_code = ErrorCode.LLM_ERROR


class ToolError(HarnessError):
    """May be raised inside a tool; the executor converts it into an error `ToolResult`."""

    default_code = ErrorCode.TOOL_ERROR


class ValidationError(HarnessError):
    default_code = ErrorCode.INVALID_ARGS


class HarnessTimeoutError(HarnessError):
    default_code = ErrorCode.TIMEOUT


class ExecutionError(HarnessError):
    default_code = ErrorCode.EXECUTION_ERROR


class ContextError(HarnessError):
    default_code = ErrorCode.CONTEXT_OVERFLOW
