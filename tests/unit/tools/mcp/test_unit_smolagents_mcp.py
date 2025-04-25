# pylint: disable=unused-argument, unused-variable, attr-de
import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


def create_mock_tools() -> list[MagicMock]:
    """Helper method to create mock tools."""
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    return [mock_tool1, mock_tool2]


def create_specific_mock_tools() -> list[MagicMock]:
    """Helper method to create specific mock tools."""
    mock_read_tool = MagicMock()
    mock_read_tool.name = "read_thing"
    mock_write_tool = MagicMock()
    mock_write_tool.name = "write_thing"
    mock_other_tool = MagicMock()
    mock_other_tool.name = "other_thing"
    return [mock_read_tool, mock_write_tool, mock_other_tool]


@patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient")
class TestSmolagentsMCPServer(unittest.TestCase):
    """Tests for the SmolagentsMCPServer class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock(spec=MCPStdioParams)
        self.test_tool.command = "test_command"
        self.test_tool.args = ["arg1", "arg2"]

    def test__setup_tools_with_none_tools(
        self,
        mock_client_class: Any,
    ) -> None:
        """Test that when mcp_tool.tools is None, all available tools are used."""
        # Setup mock tools
        mock_tools = create_mock_tools()

        # Setup mock MCPClient context manager behavior
        mock_client_class.return_value.__enter__.return_value = mock_tools

        self.test_tool.tools = None
        mcp_server = _get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server._setup_tools())

        # Verify all tools are included
        assert mcp_server.tools == mock_tools
        assert len(mcp_server.tools) == 2

    def test__setup_tools_with_specific_tools(
        self,
        mock_client_class: Any,
    ) -> None:
        """Test that when mcp_tool.tools has specific values, only those tools are used."""
        # Setup mock tools
        mock_tools = create_specific_mock_tools()

        # Setup mock MCPClient context manager behavior
        mock_client_class.return_value.__enter__.return_value = mock_tools

        # Create test tool configuration with specific tools
        self.test_tool.tools = ["read_thing", "write_thing"]

        mcp_server = _get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server._setup_tools())

        # Verify only the requested tools are included
        assert len(mcp_server.tools) == 2
        tool_names = [tool.name for tool in mcp_server.tools]  # type: ignore[union-attr]
        assert "read_thing" in tool_names
        assert "write_thing" in tool_names
        assert "other_thing" not in tool_names


@pytest.mark.asyncio
async def test_smolagents_mcp_sse() -> None:
    # Create mock tools
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    mock_tools = [mock_tool1, mock_tool2]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(url="http://localhost:8000/sse")

    # Create the server instance
    server = _get_mcp_server(mcp_tool, AgentFramework.SMOLAGENTS)

    # Patch the MCPClient class to return our mock tools
    with patch(
        "any_agent.tools.mcp.frameworks.smolagents.MCPClient"
    ) as mock_client_class:
        # Setup the mock to return our tools when used as a context manager
        mock_client_class.return_value.__enter__.return_value = mock_tools

        # Test the _setup_tools method
        await server._setup_tools()

        # Verify the client was created with correct parameters
        mock_client_class.assert_called_once_with({"url": "http://localhost:8000/sse"})

        # Verify tools were correctly assigned
        assert server.tools == mock_tools
