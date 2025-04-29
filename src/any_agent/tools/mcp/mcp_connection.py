from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, PrivateAttr

from any_agent.config import Tool

if TYPE_CHECKING:
    from agents.mcp.server import MCPServer


class MCPConnection(BaseModel, ABC):
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def list_tools(self) -> list[Tool]:
        """List tools from the MCP server."""

    @property
    def server(self) -> "MCPServer | None":
        """Return the MCP server instance."""
        return None
