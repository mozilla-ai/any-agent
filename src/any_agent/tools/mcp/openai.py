"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from contextlib import AsyncExitStack, suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams
from any_agent.logging import logger

from .mcp_server_base import MCPServerBase

with suppress(ImportError):
    from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse
    from agents.mcp import (
        MCPServerSseParams as OpenAIInternalMCPServerSseParams,
    )
    from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
    from agents.mcp import (
        MCPServerStdioParams as OpenAIInternalMCPServerStdioParams,
    )



class OpenAIMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server: OpenAIInternalMCPServerStdio | OpenAIInternalMCPServerSse | None = None
        self.exit_stack = AsyncExitStack()

    async def setup_stdio_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPStdioParams)

        params = OpenAIInternalMCPServerStdioParams(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
        )

        self.server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params=params,
        )

    async def setup_sse_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPSseParams)

        params = OpenAIInternalMCPServerSseParams(url=self.mcp_tool.url)

        self.server = OpenAIInternalMCPServerSse(
            name="OpenAI MCP Server", params=params
        )

    async def setup_tools(self) -> None:
        """Set up the OpenAI MCP server with the provided configuration."""
        await super().setup_tools()

        assert self.server

        await self.exit_stack.enter_async_context(self.server)
        # Get tools from the server
        self.tools = await self.server.list_tools()
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools",
        )
