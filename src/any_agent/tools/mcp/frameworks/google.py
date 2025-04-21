"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

# from any_agent.tools.mcp.mcp_server import MCPServer

with suppress(ImportError):
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import (
        SseServerParams as GoogleSseServerParameters,
    )
    from google.adk.tools.mcp_tool.mcp_toolset import (
        StdioServerParameters as GoogleStdioServerParameters,
    )


class GoogleMCPServer(object):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server: GoogleMCPToolset | None = None
        self.exit_stack = AsyncExitStack()
        self.params: GoogleStdioServerParameters | GoogleSseServerParameters | None = (
            None
        )

    async def setup_stdio_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPStdioParams)

        self.params = GoogleStdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

    async def setup_sse_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPSseParams)

        self.params = GoogleSseServerParameters(
            url=self.mcp_tool.url,
            headers=self.mcp_tool.headers,
        )

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        await super().setup_tools()

        assert self.params
        toolset = GoogleMCPToolset(connection_params=self.params)
        await self.exit_stack.enter_async_context(toolset)
        tools = await toolset.load_tools()
        self.tools = tools
        self.server = toolset
