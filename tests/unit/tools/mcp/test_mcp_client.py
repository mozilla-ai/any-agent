import inspect
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from any_agent.config import AgentFramework, MCPStdio
from any_agent.tools.mcp.mcp_client import MCPClient


@pytest.mark.asyncio
async def test_create_tool_function_with_complex_schema() -> None:
    """Test the core tool function creation with comprehensive parameter handling."""
    from mcp.types import Tool as MCPTool

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock()]
    mock_result.content[0].text = "Tool executed successfully"
    mock_session.call_tool.return_value = mock_result

    config = MCPStdio(command="test", args=[])
    client = MCPClient(config=config, framework=AgentFramework.OPENAI)
    client._session = mock_session

    complex_tool = MCPTool(
        name="complex_search",
        description="Search with multiple parameter types",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include result metadata",
                },
                "filters": {"type": "object", "description": "Search filters"},
                "tags": {"type": "array", "description": "Filter tags"},
                "threshold": {"type": "number", "description": "Similarity threshold"},
                "optional_param": {
                    "type": "string",
                    "description": "Optional parameter",
                },
            },
            "required": ["query", "max_results", "include_metadata"],
        },
    )

    tool_func = client._create_tool_function(complex_tool)

    assert tool_func.__name__ == "complex_search"
    assert tool_func.__doc__ is not None
    assert "Search with multiple parameter types" in tool_func.__doc__
    assert "query: Search query string" in tool_func.__doc__
    assert "max_results: Maximum number of results" in tool_func.__doc__

    sig = inspect.signature(tool_func)
    params = sig.parameters

    assert "query" in params
    assert params["query"].annotation is str
    assert params["query"].default is inspect.Parameter.empty

    assert "max_results" in params
    assert params["max_results"].annotation is int
    assert params["max_results"].default is inspect.Parameter.empty

    assert "include_metadata" in params
    assert params["include_metadata"].annotation is bool
    assert params["include_metadata"].default is inspect.Parameter.empty

    assert "optional_param" in params
    assert params["optional_param"].annotation == Optional[str]  # noqa: UP045
    assert params["optional_param"].default is None

    assert params["filters"].annotation == Optional[dict]  # noqa: UP045
    assert params["tags"].annotation == Optional[list]  # noqa: UP045
    assert params["threshold"].annotation == Optional[float]  # noqa: UP045

    assert sig.return_annotation is str

    result = await tool_func(query="test query", max_results=10, include_metadata=True)
    assert result == "Tool executed successfully"
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {"query": "test query", "max_results": 10, "include_metadata": True},
    )

    await tool_func(
        query="another query",
        max_results=5,
        include_metadata=False,
        optional_param="optional_value",
        threshold=0.8,
    )
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {
            "query": "another query",
            "max_results": 5,
            "include_metadata": False,
            "optional_param": "optional_value",
            "threshold": 0.8,
        },
    )

    client._session = None
    error_result = await tool_func(query="test", max_results=1, include_metadata=True)
    assert "Error: MCP session not available" in error_result
