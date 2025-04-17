"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import suppress

from any_agent.config import MCPTool

from .mcp_server_base import MCPServerBase

with suppress(ImportError):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client


class LangchainMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for LangChain agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)
        self.client = None
        self.session = None
        self.tools = []

    async def setup_tools(self):
        """Set up the LangChain MCP server with the provided configuration."""
        from langchain_mcp_adapters.tools import load_mcp_tools

        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )
        self.client = stdio_client(server_params)
        self.read, self.write = await self.client.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        self.tools = await load_mcp_tools(self.session)
