"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress
from typing import Any

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    mcp_available = True


class LangchainMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for LangChain agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, mcp_available, "mcp langchain-mcp-adapters")
        self.client: Any | None = None
        self.tools = []
        self.session: ClientSession = None
        self.exit_stack = AsyncExitStack()

    async def setup_stdio_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPStdioParams)

        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )

        self.client = stdio_client(server_params)

    async def setup_sse_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPSseParams)

        self.client = sse_client(
            url=self.mcp_tool.url,
            headers=self.mcp_tool.headers,
        )

    async def setup_tools(self) -> None:
        """Set up the LangChain MCP server with the provided configuration."""
        await super().setup_tools()

        assert self.client
        stdio, write = await self.exit_stack.enter_async_context(self.client)

        client_session = ClientSession(stdio, write)
        self.session = await self.exit_stack.enter_async_context(client_session)

        await self.session.initialize()
        # List available tools
        self.tools = await load_mcp_tools(self.session)
