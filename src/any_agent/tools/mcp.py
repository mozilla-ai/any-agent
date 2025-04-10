"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
import os
import asyncio
from loguru import logger
from textwrap import dedent

from any_agent.config import MCPTool

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    mcp_available = True
except ImportError:
    mcp_available = False


# Global registry to keep Smolagents MCP manager instances alive
_smolagents_mcp_managers = {}


class MCPServerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(self, mcp_tool: MCPTool):
        if not mcp_available:
            raise ImportError(
                "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            )

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools = []

    @abstractmethod
    def setup_tools(self):
        """Set up tools. To be implemented by subclasses."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources. To be implemented by subclasses."""
        pass


class SmolagentsMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        # Generate a unique identifier for this manager instance
        self.id = id(self)
        self.context = None
        self.tool_collection = None

        # Register self in the global registry to prevent garbage collection
        # (only needed for Smolagents implementation)
        _smolagents_mcp_managers[self.id] = self

        self.setup_tools()

    def setup_tools(self):
        from smolagents import ToolCollection

        self.server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )

        # Store the context manager itself
        self.context = ToolCollection.from_mcp(
            self.server_parameters, trust_remote_code=True
        )
        # Enter the context
        self.tool_collection = self.context.__enter__()
        tools = self.tool_collection.tools

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        if requested_tools:
            filtered_tools = [tool for tool in tools if tool.name in requested_tools]
            if len(filtered_tools) != len(requested_tools):
                tool_names = [tool.name for tool in filtered_tools]
                raise ValueError(
                    dedent(f"""Could not find all requested tools in the MCP server:
                                Requested: {requested_tools}
                                Set:   {tool_names}""")
                )
            self.tools = filtered_tools
        else:
            logger.info(
                "No specific tools requested for MCP server, using all available tools:"
            )
            logger.info(f"Tools available: {tools}")
            self.tools = tools

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
        # Remove from registry when deleted (Smolagents-specific)
        if hasattr(self, "id") and self.id in _smolagents_mcp_managers:
            del _smolagents_mcp_managers[self.id]


class OpenAIMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.server = None
        self.loop = None
        self.setup_tools()

    def setup_tools(self):
        """Set up the OpenAI MCP server with the provided configuration."""
        from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio

        self.server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params={
                "command": self.mcp_tool.command,
                "args": self.mcp_tool.args,
            },
        )
        try:
            # Try to get the existing event loop
            self.loop = asyncio.get_event_loop()
            # Check if it's running
            is_running = self.loop.is_running()
        except RuntimeError:
            # No event loop exists in this thread
            is_running = False
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # If we got an existing event loop but it's not running, we might need a new one
        if not is_running:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self.server.__aenter__())
        # Get tools from the server
        self.tools = self.loop.run_until_complete(self.server.list_tools())
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools"
        )

    def cleanup(self):
        self.server = None

    def __del__(self):
        self.cleanup()


class LangchainMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for LangChain agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.client = None
        self.session = None
        self.tools = []

        # Using an existing event loop if available, or creating a new one
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self.setup_tools())

    async def setup_tools(self):
        """Set up the LangChain MCP server with the provided configuration."""
        from langchain_mcp_adapters.tools import load_mcp_tools

        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )
        self.client = stdio_client(server_params)
        self.read, self.write = await self.client.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        self.tools = await load_mcp_tools(self.session)

    def cleanup(self):
        """Clean up resources using the same event loop"""
        if self.session and self.client:
            try:
                # Create a new event loop for cleanup if the original is closed
                if self.loop.is_closed():
                    cleanup_loop = asyncio.new_event_loop()
                    cleanup_loop.run_until_complete(self._cleanup_async())
                    cleanup_loop.close()
                else:
                    self.loop.run_until_complete(self._cleanup_async())
            except Exception as e:
                logger.error(f"Error closing LangChain MCP resources: {e}")

    async def _cleanup_async(self):
        """Async cleanup to be run in the same event loop as setup"""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None

        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    def __del__(self):
        self.cleanup()

    # Consider adding a context manager interface
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup_async()
