"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os

from any_agent.config import MCPParams, MCPStdioParams

from .mcp_server_base import MCPServerBase


class AgnoMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for Agno agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server = None

    async def setup_tools(self) -> None:
        """Set up the Agno MCP server with the provided configuration."""
        from agno.tools.mcp import MCPTools

        if not isinstance(self.mcp_tool, MCPStdioParams):
            raise NotImplementedError

        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self.server = MCPTools(
            command=server_params,
            include_tools=list(self.mcp_tool.tools or []),
            env={**os.environ},
        )
        self.tools = await self.server.__aenter__()  # type: ignore[attr-defined]
