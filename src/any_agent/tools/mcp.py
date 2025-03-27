"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
import os
from loguru import logger
from any_agent.schema import MCPTool

# Global registry to keep manager instances alive
_mcp_managers = {}


class MCPToolsManagerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(self, mcp_tool: MCPTool):
        # Generate a unique identifier for this manager instance
        self.id = id(self)

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools = []

        # Register self in the global registry to prevent garbage collection
        _mcp_managers[self.id] = self

    def __del__(self):
        # Remove from registry when deleted
        if self.id in _mcp_managers:
            del _mcp_managers[self.id]

    @abstractmethod
    def setup_tools(self):
        """Set up tools. To be implemented by subclasses."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources. To be implemented by subclasses."""
        pass


class SmolagentsMCPToolsManager(MCPToolsManagerBase):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.context = None
        self.tool_collection = None
        self.setup_tools()

    def setup_tools(self):
        from mcp import StdioServerParameters
        from smolagents import ToolCollection

        self.server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )

        # Store the context manager itself
        self.context = ToolCollection.from_mcp(self.server_parameters)
        # Enter the context
        self.tool_collection = self.context.__enter__()
        self.tools = self.tool_collection.tools

    def cleanup(self):
        # Exit the context when cleanup is called
        if hasattr(self, "context") and self.context:
            try:
                self.context.__exit__(None, None, None)
                self.context = None
            except Exception as e:
                logger.error(f"Error closing MCP context: {e}")

    def __del__(self):
        self.cleanup()
        super().__del__()
