"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from any_agent.config import MCPParams, MCPStdioParams

from .mcp_server_base import MCPServerBase

if TYPE_CHECKING:
    from collections.abc import Callable


with suppress(ImportError):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client


class LangchainMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for LangChain agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.client = None
        self.session = None
        self.tools = []
        self.read: Callable[..., Any] | None = None
        self.write: Callable[..., Any] | None = None

    async def setup_tools(self) -> None:
        """Set up the LangChain MCP server with the provided configuration."""
        from langchain_mcp_adapters.tools import load_mcp_tools

        if not isinstance(self.mcp_tool, MCPStdioParams):
            raise NotImplementedError

        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.client = stdio_client(server_params)
        self.read, self.write = await self.client.__aenter__()  # type: ignore[attr-defined]
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()  # type: ignore[attr-defined]
        await self.session.initialize()  # type: ignore[attr-defined]
        self.tools = await load_mcp_tools(self.session)
