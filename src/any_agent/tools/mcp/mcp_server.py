"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, Field

from any_agent.config import Tool

from .frameworks import MCPFrameworkConnection


class MCPServer(BaseModel):
    """Base class for MCP tools managers across different frameworks."""

    mcp_connection: MCPFrameworkConnection
    tools: Sequence[Tool] = Field(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        self.mcp_connection.check_dependencies()

    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""
        self.tools = await self.mcp_connection.setup()
