# pylint: disable=unused-argument, unused-variable
# Test MCP Tools Classes.
# Disclaim

import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools import get_mcp_server


@patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient")
class TestSmolagentsMCPServer(unittest.TestCase):
    """Tests for the SmolagentsMCPServer class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock(spec=MCPStdioParams)
        self.test_tool.command = "test_command"
        self.test_tool.args = ["arg1", "arg2"]

    def test_setup_tools_with_none_tools(
        self,
        mock_client_class: Any,
    ) -> None:
        """Test that when mcp_tool.tools is None, all available tools are used."""
        # Setup mock tools
        mock_tools = create_mock_tools()

        # Setup mock MCPClient context manager behavior
        mock_client_class.return_value.__enter__.return_value = mock_tools

        self.test_tool.tools = None
        mcp_server = get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify all tools are included
        assert mcp_server.tools == mock_tools
        assert len(mcp_server.tools) == 2

    def test_setup_tools_with_specific_tools(
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

        mcp_server = get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

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
    server = get_mcp_server(mcp_tool, AgentFramework.SMOLAGENTS)

    # Patch the MCPClient class to return our mock tools
    with patch(
        "any_agent.tools.mcp.frameworks.smolagents.MCPClient"
    ) as mock_client_class:
        # Setup the mock to return our tools when used as a context manager
        mock_client_class.return_value.__enter__.return_value = mock_tools

        # Test the setup_tools method
        await server.setup_tools()

        # Verify the client was created with correct parameters
        mock_client_class.assert_called_once_with({"url": "http://localhost:8000/sse"})

        # Verify tools were correctly assigned
        assert server.tools == mock_tools
