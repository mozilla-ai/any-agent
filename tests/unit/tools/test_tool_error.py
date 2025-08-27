"""Tests for ToolError implementation."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from any_agent.tools.wrappers import ToolError, _wrap_no_exception
from any_agent.config import AgentFramework
from any_agent.callbacks.span_generation.base import _SpanGeneration
from opentelemetry.trace import Status, StatusCode


def test_tool_error_creation():
    """Test that ToolError is created with correct attributes."""
    error = ToolError(
        message="Test error",
        error_type="ValueError",
        tool_name="test_tool",
        framework=AgentFramework.TINYAGENT,
        timestamp="2024-01-01T00:00:00"
    )
    
    assert error.message == "Test error"
    assert error.error_type == "ValueError"
    assert error.tool_name == "test_tool"
    assert error.framework == AgentFramework.TINYAGENT
    assert error.timestamp == "2024-01-01T00:00:00"
    assert error.traceback is None


def test_tool_error_string_representation():
    """Test backward-compatible string representation."""
    error = ToolError(
        message="Something went wrong",
        error_type="RuntimeError",
        tool_name="my_tool",
        framework=AgentFramework.OPENAI,
        timestamp=datetime.now().isoformat()
    )
    
    # Should maintain backward compatibility
    assert str(error) == "Error calling tool: Something went wrong"


def test_wrap_no_exception_with_sync_tool():
    """Test wrapping synchronous tool that raises exception."""
    def failing_tool(x: int) -> str:
        """A tool that always fails."""
        raise ValueError("Test error")
    
    wrapped = _wrap_no_exception(failing_tool, AgentFramework.LANGCHAIN)
    result = wrapped(42)
    
    assert isinstance(result, ToolError)
    assert result.message == "Test error"
    assert result.error_type == "ValueError"
    assert result.tool_name == "failing_tool"
    assert result.framework == AgentFramework.LANGCHAIN


@pytest.mark.asyncio
async def test_wrap_no_exception_with_async_tool():
    """Test wrapping asynchronous tool that raises exception."""
    async def async_failing_tool(x: int) -> str:
        """An async tool that always fails."""
        raise RuntimeError("Async error")
    
    wrapped = _wrap_no_exception(async_failing_tool, AgentFramework.LLAMA_INDEX)
    result = await wrapped(42)
    
    assert isinstance(result, ToolError)
    assert result.message == "Async error"
    assert result.error_type == "RuntimeError"
    assert result.tool_name == "async_failing_tool"
    assert result.framework == AgentFramework.LLAMA_INDEX


def test_span_generation_detects_tool_error():
    """Test that span generation correctly detects ToolError."""
    tool_error = ToolError(
        message="Tool failed",
        error_type="Exception",
        tool_name="my_tool",
        framework=AgentFramework.TINYAGENT,
        timestamp="2024-01-01T00:00:00"
    )
    
    span_gen = _SpanGeneration()
    status = span_gen._determine_tool_status(tool_error, "text")
    
    assert isinstance(status, Status)
    assert status.status_code == StatusCode.ERROR
    assert "Tool 'my_tool' failed - Exception: Tool failed" in status.description


def test_span_generation_backward_compatibility():
    """Test backward compatibility with string error formats."""
    span_gen = _SpanGeneration()
    
    # Test "Error calling tool:" pattern
    error1 = "Error calling tool: Something broke"
    status1 = span_gen._determine_tool_status(error1, "text")
    assert status1.status_code == StatusCode.ERROR
    assert status1.description == error1
    
    
    # Test normal output (not an error)
    normal = "Successfully fetched data"
    status3 = span_gen._determine_tool_status(normal, "text")
    assert status3 == StatusCode.OK


def test_tool_error_in_set_tool_output():
    """Test _set_tool_output with ToolError."""
    context = MagicMock()
    span_gen = _SpanGeneration()
    
    tool_error = ToolError(
        message="API call failed", 
        error_type="HTTPError",
        tool_name="fetch_data",
        framework=AgentFramework.GOOGLE,
        timestamp="2024-01-01T00:00:00"
    )
    
    with patch("any_agent.callbacks.span_generation.base.Status") as status_mock:
        span_gen._set_tool_output(context, tool_error)
        
        # Verify attributes were set with serialized error
        context.current_span.set_attributes.assert_called_once()
        attrs = context.current_span.set_attributes.call_args[0][0]
        assert "Error calling tool: API call failed" in attrs["gen_ai.output"]
        
        # Verify error status was set
        context.current_span.set_status.assert_called_once()