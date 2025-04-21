"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING

from pydantic import BaseModel

from any_agent.config import MCPParams, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence


mcp_available = False
with suppress(ImportError):
    mcp_available = True

class MCPToolConnection(BaseModel, ABC):
    mcp_tool: MCPParams

    @abstractmethod
    def setup(self) -> None:
        ...


class MCPServerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(self, mcp_tool: MCPParams):
        if not mcp_available:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            raise ImportError(msg)

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools: Sequence[Tool] = []

    @property
    @abstractmethod
    def mcp_tool_connection(self) -> MCPToolConnection:
        ...

    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
        self.tools = await self.mcp_tool_connection.setup()
