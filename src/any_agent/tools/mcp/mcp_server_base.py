"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from contextlib import suppress
from typing import TYPE_CHECKING

from pydantic import BaseModel

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams, Tool

if TYPE_CHECKING:
    from collections.abc import Sequence


mcp_available = False
with suppress(ImportError):
    mcp_available = True

class MCPToolType(str, Enum):
    STDIO = auto()
    SSE = auto()


class MCPToolConnection(BaseModel, ABC):
    mcp_tool: MCPParams

    @abstractmethod
    def setup(self) -> None:
        ...

def mcp_params_to_mcp_tool_type(mcp_params: MCPParams) -> MCPToolType:
    """Convert MCP parameters to a tool type."""
    match mcp_params:
        case MCPStdioParams():
            return MCPToolType.STDIO
        case MCPSseParams():
            return MCPToolType.SSE

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
