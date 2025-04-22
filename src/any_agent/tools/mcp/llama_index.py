"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import suppress

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
    from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

    mcp_available = True


class LlamaIndexMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, "any-agent[mcp,llama_index]", mcp_available)
        self.client: LlamaIndexMCPClient | None = None

    async def setup_stdio_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPStdioParams):
            msg = "MCP tool parameters must be of type MCPStdioParams for stdio server."
            raise ValueError(msg)
        self.client = LlamaIndexMCPClient(
            command_or_url=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )

    async def setup_sse_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPSseParams):
            msg = "MCP tool parameters must be of type MCPSseParams for SSE server."
            raise ValueError(msg)
        self.client = LlamaIndexMCPClient(command_or_url=self.mcp_tool.url)

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        await super().setup_tools()

        if not self.client:
            msg = "MCP client is not set up. Please call setup_stdio_tools or setup_sse_tools first."
            raise ValueError(msg)
        mcp_tool_spec = LlamaIndexMcpToolSpec(
            client=self.client,
            allowed_tools=self.mcp_tool.tools,
        )

        self.tools = await mcp_tool_spec.to_tool_list_async()
