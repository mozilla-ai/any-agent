"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from agno.tools.mcp import MCPTools as AgnoMCPTools
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    mcp_available = True


class AgnoMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Agno agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, "any-agent[mcp,agno]", mcp_available)
        self.exit_stack = AsyncExitStack()
        self.server: AgnoMCPTools | None = None

    async def setup_stdio_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPStdioParams):
            msg = "MCP tool parameters must be of type MCPStdioParams for stdio server."
            raise ValueError(msg)

        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self.server = AgnoMCPTools(
            command=server_params,
            include_tools=self.mcp_tool.tools,
            env={**os.environ},
        )

    async def setup_sse_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPSseParams):
            msg = "MCP tool parameters must be of type MCPSseParams for SSE server."
            raise ValueError(msg)
        client = sse_client(
            url=self.mcp_tool.url,
            headers=self.mcp_tool.headers,
        )
        sse_transport = await self.exit_stack.enter_async_context(client)
        stdio, write = sse_transport
        client_session = ClientSession(stdio, write)
        session = await self.exit_stack.enter_async_context(client_session)
        await session.initialize()
        self.server = AgnoMCPTools(
            session=session,
            include_tools=self.mcp_tool.tools,
        )

    async def setup_tools(self) -> None:
        """Set up the Agno MCP server with the provided configuration."""
        await super().setup_tools()

        if not self.server:
            msg = "MCP server is not set up. Please call setup_stdio_tools or setup_sse_tools first."
            raise ValueError(msg)

        self.tools = [await self.exit_stack.enter_async_context(self.server)]
