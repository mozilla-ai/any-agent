# Test MCP Tools Classes.
# Disclaim

import asyncio
import unittest
from unittest.mock import patch, MagicMock

from any_agent.tools.mcp import (
    MCPServerBase,
    SmolagentsMCPServerStdio,
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
        pass


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
class TestSmolagentsMCPServerStdio(unittest.TestCase):
    """Tests for the SmolagentsMCPServerStdio class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock()
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

        mcp_server = SmolagentsMCPServerStdio(self.test_tool)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify all tools are included
        self.assertEqual(mcp_server.tools, mock_tools)
        self.assertEqual(len(mcp_server.tools), 2)

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

        mcp_server = SmolagentsMCPServerStdio(self.test_tool)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify only the requested tools are included
        self.assertEqual(len(mcp_server.tools), 2)
        tool_names = [tool.name for tool in mcp_server.tools]
        self.assertIn("read_thing", tool_names)
        self.assertIn("write_thing", tool_names)
        self.assertNotIn("other_thing", tool_names)
