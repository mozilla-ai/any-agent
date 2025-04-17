"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from importlib.util import find_spec

from any_agent.config import MCPTool

client_session_available = find_spec("mcp") is not None
stdio_client_available = find_spec("mcp.client.stdio") is not None
mcp_available = client_session_available and stdio_client_available


class MCPServerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(self, mcp_tool: MCPTool):
        if not mcp_available:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            raise ImportError(msg)

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools = []

    @abstractmethod
    async def setup_tools(self):
        """Set up tools. To be implemented by subclasses."""
