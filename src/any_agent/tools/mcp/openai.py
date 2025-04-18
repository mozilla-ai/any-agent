"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from any_agent.config import MCPParams, MCPStdioParams
from any_agent.logging import logger

from .mcp_server_base import MCPServerBase


class OpenAIMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server = None
        self.loop = None

    async def setup_tools(self) -> None:
        """Set up the OpenAI MCP server with the provided configuration."""
        from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio

        if not isinstance(self.mcp_tool, MCPStdioParams):
            raise NotImplementedError

        self.server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params={
                "command": self.mcp_tool.command,
                "args": list(self.mcp_tool.args),
            },
        )

        await self.server.__aenter__()  # type: ignore[attr-defined]
        # Get tools from the server
        self.tools = await self.server.list_tools()  # type: ignore[attr-defined]
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools"
        )
