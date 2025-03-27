import unittest
from unittest.mock import patch, MagicMock

from any_agent.tools.mcp import (
    MCPToolsManagerBase,
    SmolagentsMCPToolsManager,
    _mcp_managers,
)


class TestMCPToolsManagerBase(unittest.TestCase):
    """Tests for the MCPToolsManagerBase class."""

    def test_init_registers_in_global_registry(self):
        """Test that initialization registers the instance in the global registry."""

        # Create a concrete subclass for testing the abstract base class
        class ConcreteMCPManager(MCPToolsManagerBase):
            def setup_tools(self):
                pass

            def cleanup(self):
                pass

        test_tool = {"name": "test_tool"}
        manager = ConcreteMCPManager(test_tool)

        # Check if manager is registered in global registry
        self.assertIn(manager.id, _mcp_managers)
        self.assertEqual(_mcp_managers[manager.id], manager)

        # Check if attributes are set correctly
        self.assertEqual(manager.mcp_tool, test_tool)
        self.assertEqual(manager.tools, [])

    def test_del_removes_from_registry(self):
        """Test that deletion removes the instance from the global registry."""

        class ConcreteMCPManager(MCPToolsManagerBase):
            def setup_tools(self):
                pass

            def cleanup(self):
                pass

        manager = ConcreteMCPManager({"name": "test_tool"})
        manager_id = manager.id

        # Verify it's in the registry
        self.assertIn(manager_id, _mcp_managers)

        # Delete the manager
        manager.__del__()

        # Verify it's removed from the registry
        self.assertNotIn(manager_id, _mcp_managers)


class TestSmolagentsMCPToolsManager(unittest.TestCase):
    """Tests for the SmolagentsMCPToolsManager class."""

    @patch("smolagents.ToolCollection")
    @patch("mcp.StdioServerParameters")
    def test_cleanup(self, mock_stdio_params, mock_tool_collection):
        """Test that cleanup exits the context manager."""
        # Mock the context manager behavior
        mock_context = MagicMock()
        mock_tool_collection.from_mcp.return_value = mock_context

        # Create test tool configuration
        test_tool = {"command": "test_command", "args": ["arg1", "arg2"]}

        # Initialize the manager
        with patch.object(SmolagentsMCPToolsManager, "setup_tools", return_value=None):
            manager = SmolagentsMCPToolsManager(test_tool)

        # Set the context attribute
        manager.context = mock_context

        # Call cleanup
        manager.cleanup()

        # Verify context.__exit__ was called
        mock_context.__exit__.assert_called_once_with(None, None, None)

        # Verify context was set to None
        self.assertIsNone(manager.context)
