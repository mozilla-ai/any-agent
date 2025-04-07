# Test MCP Tools Classes.
# Disclaim

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

    def test_cleanup(self, mock_stdio_params, mock_tool_collection):
        """Test that cleanup exits the context manager."""
        # Configure the mocks
        mock_tool_collection.from_mcp.return_value = self.mock_context

        # Initialize the manager with setup_tools patched
        with patch.object(SmolagentsMCPServerStdio, "setup_tools", return_value=None):
            manager = SmolagentsMCPServerStdio(self.test_tool)

        # Set the context attribute
        manager.context = self.mock_context

        # Call cleanup
        manager.cleanup()

        # Verify context.__exit__ was called
        self.mock_context.__exit__.assert_called_once_with(None, None, None)

        # Verify context was set to None
        self.assertIsNone(manager.context)

    def test_setup_tools_with_none_tools(self, mock_stdio_params, mock_tool_collection):
        """Test that when mcp_tool.tools is None, all available tools are used."""
        # Setup mock tools
        mock_tools = create_mock_tools()

        # Setup mock tool collection
        self.mock_collection.tools = mock_tools
        mock_tool_collection.from_mcp.return_value = self.mock_context

        # Create test tool configuration with None tools
        self.test_tool.tools = None

        # Initialize the manager
        manager = SmolagentsMCPServerStdio(self.test_tool)

        # Verify all tools are included
        self.assertEqual(manager.tools, mock_tools)
        self.assertEqual(len(manager.tools), 2)

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

        # Initialize the manager
        manager = SmolagentsMCPServerStdio(self.test_tool)

        # Verify only the requested tools are included
        self.assertEqual(len(manager.tools), 2)
        tool_names = [tool.name for tool in manager.tools]
        self.assertIn("read_thing", tool_names)
        self.assertIn("write_thing", tool_names)
        self.assertNotIn("other_thing", tool_names)
