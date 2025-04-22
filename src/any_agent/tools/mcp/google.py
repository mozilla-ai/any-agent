"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import (
        SseServerParams as GoogleSseServerParameters,
    )
    from google.adk.tools.mcp_tool.mcp_toolset import (
        StdioServerParameters as GoogleStdioServerParameters,
    )

    mcp_available = True


class GoogleMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, "any-agent[mcp,google]", mcp_available)
        self.server: GoogleMCPToolset | None = None
        self.exit_stack = AsyncExitStack()
        self.params: GoogleStdioServerParameters | GoogleSseServerParameters | None = (
            None
        )

    async def setup_stdio_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPStdioParams):
            msg = "MCP tool parameters must be of type MCPStdioParams for stdio server."
            raise ValueError(msg)

        self.params = GoogleStdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

    async def setup_sse_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPSseParams):
            msg = "MCP tool parameters must be of type MCPSseParams for SSE server."
            raise ValueError(msg)

        self.params = GoogleSseServerParameters(
            url=self.mcp_tool.url,
            headers=self.mcp_tool.headers,
        )

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        await super().setup_tools()

        if not self.params:
            msg = "MCP server parameters are not set up. Please call setup_stdio_tools or setup_sse_tools first."
            raise ValueError(msg)
        toolset = GoogleMCPToolset(connection_params=self.params)
        await self.exit_stack.enter_async_context(toolset)
        tools = await toolset.load_tools()
        self.tools = tools
        self.server = toolset
