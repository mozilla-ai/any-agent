# pylint: disable=unused-argument, unused-variable
from collections.abc import Sequence
from typing import Any

import pytest

from any_agent.config import AgentFramework, MCPParams, MCPSse, Tool
from any_agent.tools import _wrap_tools, MCPClient


@pytest.mark.asyncio
async def test_mcp_tool_wrapping(
    agent_framework: AgentFramework,
    tools: Sequence[Tool],
) -> None:
    """Test that MCP tools are properly wrapped for different frameworks."""
    from unittest.mock import AsyncMock, patch

    # Create proper mock functions with docstrings for framework compatibility
    def create_mock_tool(tool_name: str) -> Any:
        def mock_tool() -> str:
            """Mock tool for testing."""
            return f"mock_result_{tool_name}"

        mock_tool.__name__ = f"mock_tool_{tool_name}"
        mock_tool.__doc__ = f"Mock tool for {tool_name}."
        return mock_tool

    # Create a mock MCP client
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(
        return_value=[create_mock_tool(str(tool)) for tool in tools]
    )

    # Mock both MCPClient and SmolagentsMCPClient constructors to return our mock
    with (
        patch("any_agent.tools.wrappers.MCPClient", return_value=mock_client),
        patch("any_agent.tools.wrappers.SmolagentsMCPClient", return_value=mock_client),
    ):
        mcp_config = MCPSse(
            url="http://localhost:8000/sse", tools=[str(tool) for tool in tools]
        )
        wrapped_tools, mcp_clients = await _wrap_tools([mcp_config], agent_framework)

        # Should have wrapped tools and the MCP client
        assert len(mcp_clients) == 1
        assert len(wrapped_tools) >= len(tools)
        mock_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_client_tool_filtering(
    mcp_client: MCPClient,
    tools: Sequence[str],
) -> None:
    """Test the _filter_tools method of MCPClient."""
    from mcp.types import Tool as MCPTool

    # Create MCP tools with names
    all_tools = [
        MCPTool(name=tool, inputSchema={"type": "object", "properties": {}})
        for tool in ["tool1", "tool2", "tool3"]
    ]

    # Case 1: No tools requested, all tools returned
    mcp_client.config = mcp_client.config.model_copy(update={"tools": None})
    filtered_tools = mcp_client._filter_tools(all_tools)
    assert len(filtered_tools) == 3

    # Case 2: Some tools requested, only those tools returned in the right order
    mcp_client.config = mcp_client.config.model_copy(
        update={"tools": ["tool3", "tool1"]}
    )
    filtered_tools = mcp_client._filter_tools(all_tools)
    assert len(filtered_tools) == 2
    assert filtered_tools[0].name == "tool3"
    assert filtered_tools[1].name == "tool1"

    # Case 3: Request for non-existent tool raises ValueError
    mcp_client.config = mcp_client.config.model_copy(
        update={"tools": ["tool1", "nonexistent"]}
    )
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        mcp_client._filter_tools(all_tools)
    assert "Missing: ['nonexistent']" in str(excinfo.value)


def test_mcp_client_tool_conversion(
    mcp_client: MCPClient,
) -> None:
    """Test converting MCP tools to callable functions."""
    from mcp.types import Tool as MCPTool

    # Create a test MCP tool
    test_tool = MCPTool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"},
                "param2": {"type": "integer", "description": "Another parameter"},
            },
            "required": ["param1"],
        },
    )

    # Convert to callable using the fake client
    callable_tools = mcp_client._convert_tools_to_callables([test_tool])
    assert len(callable_tools) == 1

    # For the fake client, we just check that it returns a callable
    tool_func = callable_tools[0]
    assert callable(tool_func)

    # Test that it can be called (fake implementation returns a mock result)
    result = tool_func(param1="test")
    assert "test_tool" in result
