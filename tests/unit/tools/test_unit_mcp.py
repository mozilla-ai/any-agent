# Test MCP Tools Classes.
# Disclaim

import asyncio
import unittest
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentConfig, AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools.mcp import (
    MCPServerBase,
    MCPTypes,
    SmolagentsMCPServer,
)


class TestMCPServerBase(unittest.TestCase):
    """Tests for the MCPServerBase class."""

    # Define the test class once at class level instead of in each test method
    class ConcreteMCPManager(MCPServerBase):
        def setup_tools(self):
            pass

        def cleanup(self):
            pass

    def setUp(self):
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock()
        self.test_tool.name = "test_tool"

    def tearDown(self):
        """Clean up after each test."""


# Common helper functions for all test classes
def create_mock_tools():
    """Helper method to create mock tools."""
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    return [mock_tool1, mock_tool2]


def create_specific_mock_tools():
    """Helper method to create specific mock tools."""
    mock_read_tool = MagicMock()
    mock_read_tool.name = "read_thing"
    mock_write_tool = MagicMock()
    mock_write_tool.name = "write_thing"
    mock_other_tool = MagicMock()
    mock_other_tool.name = "other_thing"
    return [mock_read_tool, mock_write_tool, mock_other_tool]


@patch("smolagents.ToolCollection")
@patch("mcp.StdioServerParameters")
class TestSmolagentsMCPServer(unittest.TestCase):
    """Tests for the SmolagentsMCPServer class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock(spec=MCPStdioParams)
        self.test_tool.command = "test_command"
        self.test_tool.args = ["arg1", "arg2"]

        # Common mock configuration
        self.mock_collection = MagicMock()
        self.mock_context = MagicMock()
        self.mock_context.__enter__.return_value = self.mock_collection

    def test_setup_tools_with_none_tools(self, mock_stdio_params, mock_tool_collection):
        """Test that when mcp_tool.tools is None, all available tools are used."""
        # Setup mock tools
        mock_tools = create_mock_tools()

        # Setup mock tool collection
        self.mock_collection.tools = mock_tools
        mock_tool_collection.from_mcp.return_value = self.mock_context

        # Create test tool configuration with None tools
        self.test_tool.tools = None

        mcp_server = SmolagentsMCPServer(self.test_tool)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify all tools are included
        assert mcp_server.tools == mock_tools
        assert len(mcp_server.tools) == 2

    def test_setup_tools_with_specific_tools(
        self, mock_stdio_params, mock_tool_collection
    ):
        """Test that when mcp_tool.tools has specific values, only those tools are used."""
        # Setup mock tools
        mock_tools = create_specific_mock_tools()

        # Setup mock tool collection
        self.mock_collection.tools = mock_tools
        mock_tool_collection.from_mcp.return_value = self.mock_context

        # Create test tool configuration with specific tools
        self.test_tool.tools = ["read_thing", "write_thing"]

        mcp_server = SmolagentsMCPServer(self.test_tool)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify only the requested tools are included
        assert len(mcp_server.tools) == 2
        tool_names = [tool.name for tool in mcp_server.tools]
        assert "read_thing" in tool_names
        assert "write_thing" in tool_names
        assert "other_thing" not in tool_names


def test_openai_mcpsse():
    agent_framework = AgentFramework.OPENAI
    # Set up our mocks
    mock_server = AsyncMock()
    from mcp import Tool as MCPTool

    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_server._tools_list = [mock_tool]

    # Path the imports and class
    with patch("agents.mcp.MCPServerSse", return_value=mock_server):
        # Set up tools config for agent
        tools = [
            {"url": "http://localhost:8000/sse"},  # dummy url since it's mocked
        ]

        # Create and run the agent
        agent_config = AgentConfig(model_id="gpt-4o", tools=tools)

        agent = AnyAgent.create(agent_framework, agent_config)
        assert len(agent.tools) > 0


@pytest.mark.asyncio
async def test_smolagents_mcp_sse():
    # Mock the necessary components
    mock_tool_collection = MagicMock()
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"

    # Set up the mock tools list that will be returned by the ToolCollection
    mock_tool_collection.tools = [mock_tool1, mock_tool2]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(url="http://localhost:8000/sse", type=MCPTypes.SSE)

    # Create the server instance
    server = SmolagentsMCPServer(mcp_tool)

    # Patch the ToolCollection.from_mcp method to return our mock
    with patch("smolagents.ToolCollection") as mock_tool_collection:
        # Mock the context manager behavior
        mock_tool_collection.from_mcp.return_value.__enter__.return_value = (
            mock_tool_collection
        )

        # Mock AsyncExitStack to avoid actually setting up exit handlers
        with patch.object(AsyncExitStack, "enter_context") as mock_enter_context:
            mock_enter_context.return_value = mock_tool_collection

            # Test the setup_tools method
            await server.setup_tools()

            # Verify the SSE server parameters were created correctly
            mock_tool_collection.from_mcp.assert_called_once()
            args, kwargs = mock_tool_collection.from_mcp.call_args

            # Check that the right params were passed
            assert isinstance(args[0], dict)
            assert args[0]["url"] == "http://localhost:8000/sse"
