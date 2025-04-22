"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from contextlib import AsyncExitStack, suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams
from any_agent.logging import logger

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse
    from agents.mcp import (
        MCPServerSseParams as OpenAIInternalMCPServerSseParams,
    )
    from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
    from agents.mcp import (
        MCPServerStdioParams as OpenAIInternalMCPServerStdioParams,
    )

    mcp_available = True


class OpenAIMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, "any-agent[mcp,openai]", mcp_available)
        self.server: (
            OpenAIInternalMCPServerStdio | OpenAIInternalMCPServerSse | None
        ) = None
        self.exit_stack = AsyncExitStack()

    async def setup_stdio_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPStdioParams):
            msg = "MCP tool parameters must be of type MCPStdioParams for stdio server."
            raise ValueError(msg)

        params = OpenAIInternalMCPServerStdioParams(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
        )

        self.server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params=params,
        )

    async def setup_sse_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPSseParams):
            msg = "MCP tool parameters must be of type MCPSseParams for SSE server."
            raise ValueError(msg)

        params = OpenAIInternalMCPServerSseParams(url=self.mcp_tool.url)

        self.server = OpenAIInternalMCPServerSse(
            name="OpenAI MCP Server", params=params
        )

    async def setup_tools(self) -> None:
        """Set up the OpenAI MCP server with the provided configuration."""
        await super().setup_tools()

        if not self.server:
            msg = "MCP server is not set up. Please call setup_stdio_tools or setup_sse_tools first."
            raise ValueError(msg)

        await self.exit_stack.enter_async_context(self.server)
        # Get tools from the server
        self.tools = await self.server.list_tools()
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools",
        )
