"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os

from any_agent.config import MCPTool

from .mcp_server_base import MCPServerBase


class AgnoMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for Agno agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.server = None

    async def setup_tools(self):
        """Set up the Agno MCP server with the provided configuration."""
        from agno.tools.mcp import MCPTools

        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self.server = MCPTools(
            command=server_params,
            include_tools=self.mcp_tool.tools,
            env={**os.environ},
        )
        self.tools = await self.server.__aenter__()
