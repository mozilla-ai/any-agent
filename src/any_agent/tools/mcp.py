"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
import os
import asyncio
from loguru import logger
from textwrap import dedent
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
        tools = self.tool_collection.tools

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        filtered_tools = [tool for tool in tools if tool.name in requested_tools]
        if len(filtered_tools) != len(requested_tools):
            tool_names = [tool.name for tool in filtered_tools]
            raise ValueError(
                dedent(f"""Could not find all requested tools in the MCP server:
                            Requested: {requested_tools}
                            Set:   {tool_names}""")
            )
        self.tools = filtered_tools

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


class OpenAIMCPToolsManager(MCPToolsManagerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.server = None
        self.loop = None
        self.setup_tools()

    def setup_tools(self):
        """Set up the OpenAI MCP server with the provided configuration."""
        from agents.mcp import MCPServerStdio

        self.server = MCPServerStdio(
            name="OpenAI MCP Server",
            params={
                "command": self.mcp_tool.command,
                "args": self.mcp_tool.args,
            },
        )
        # Create event loop if needed
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        # Start the server
        self.loop.run_until_complete(self.server.__aenter__())
        # Get tools from the server
        self.tools = self.loop.run_until_complete(self.server.list_tools())

    def cleanup(self):
        """Clean up the MCP server resources."""
        if self.server and self.loop:
            try:
                self.loop.run_until_complete(self.server.__aexit__(None, None, None))
                self.server = None
            except Exception as e:
                logger.error(f"Error closing OpenAI MCP server: {e}")

    def __del__(self):
        self.cleanup()
        super().__del__()
