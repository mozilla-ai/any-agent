"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

# from any_agent.tools.mcp.mcp_server import MCPServer

with suppress(ImportError):
    from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
    from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec


class LlamaIndexMCPServer:
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.client: LlamaIndexMCPClient | None = None

    async def setup_stdio_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPStdioParams)
        self.client = LlamaIndexMCPClient(
            command_or_url=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )

    async def setup_sse_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPSseParams)
        self.client = LlamaIndexMCPClient(command_or_url=self.mcp_tool.url)

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        await super().setup_tools()

        assert self.client
        mcp_tool_spec = LlamaIndexMcpToolSpec(
            client=self.client,
            allowed_tools=self.mcp_tool.tools,
        )

        self.tools = await mcp_tool_spec.to_tool_list_async()
