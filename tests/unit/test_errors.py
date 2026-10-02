from agent_harness import ErrorCode, ErrorInfo, HarnessError, LLMError, ToolError, ValidationError


def test_default_codes():
    assert LLMError("x").code == ErrorCode.LLM_ERROR
    assert ToolError("x").code == ErrorCode.TOOL_ERROR
    assert ValidationError("x").code == ErrorCode.INVALID_ARGS
    assert HarnessError("x").code == ErrorCode.UNKNOWN


def test_error_info_from_harness_error():
    info = ErrorInfo.from_exception(LLMError("slow down", code=ErrorCode.RATE_LIMIT, retryable=True, details={"s": 2}))
    assert info.model_dump() == {
        "code": "RATE_LIMIT",
        "message": "slow down",
        "type": "LLMError",
        "retryable": True,
        "details": {"s": 2},
    }


def test_error_info_from_foreign_exception_hides_traceback():
    info = ErrorInfo.from_exception(KeyError("k"))
    assert (info.code, info.type, info.retryable) == ("UNKNOWN", "KeyError", False)
    assert "Traceback" not in info.message
