"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from contextlib import suppress
from typing import Any
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams, Tool


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

def mcp_params_to_mcp_tool_type(mcp_params: Mapping[str, MCPParams]) -> str:
    """Convert MCP parameters to a tool type."""
    match mcp_params["mcp_tool"]:
        case MCPStdioParams():
            return MCPToolType.STDIO.name
        case MCPSseParams():
            return MCPToolType.SSE.name

class MCPServerBase(BaseModel, ABC):
    """Base class for MCP tools managers across different frameworks."""
    mcp_tool: MCPToolConnection
    tools: Sequence[Tool] = Field(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        if not mcp_available:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            raise ImportError(msg)

    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
        self.tools = await self.mcp_tool.setup()
