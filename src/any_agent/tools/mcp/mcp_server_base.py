"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence


class MCPServerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(
        self,
        mcp_tool: MCPParams,
        libraries: str = "any-agent[mcp]",
        mcp_available: bool = False,
    ) -> None:
        """
        Log the appropriate error if the libraries aren't installed, and set up the shared MCP variables.
        Args:
            mcp_tool (MCPParams): The MCP tool parameters.
            libraries (str): The libraries required for MCP. Used to inform error messages.
            mcp_available (bool): Flag indicating if MCP is available.
        """
        if not mcp_available:
            msg = f"You need to `pip install '{libraries}'` to use MCP."
            raise ImportError(msg)

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools: Sequence[Tool] = []

    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
        match self.mcp_tool:
            case MCPStdioParams():
                await self.setup_stdio_tools()
            case MCPSseParams():
                await self.setup_sse_tools()

    @abstractmethod
    async def setup_sse_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""

    @abstractmethod
    async def setup_stdio_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
