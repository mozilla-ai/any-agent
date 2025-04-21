"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from contextlib import suppress
from typing import Any
from collections.abc import Sequence

from pydantic import BaseModel, Field

from any_agent.config import Tool
from .frameworks import MCPFrameworkConnection

mcp_available = False
with suppress(ImportError):
    mcp_available = True


class MCPServer(BaseModel):
    """Base class for MCP tools managers across different frameworks."""
    mcp_connection: MCPFrameworkConnection
    tools: Sequence[Tool] = Field(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        if not mcp_available:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            raise ImportError(msg)

    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
        self.tools = await self.mcp_connection.setup()
