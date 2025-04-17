"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os

from any_agent.config import MCPTool

from .mcp_server_base import MCPServerBase


class LlamaIndexMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPTool):
        super().__init__(mcp_tool)

    async def setup_tools(self):
        """Set up the Google MCP server with the provided configuration."""
        from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
        from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

        mcp_client = LlamaIndexMCPClient(
            command_or_url=self.mcp_tool.command,
            args=self.mcp_tool.args,
            env={**os.environ},
        )
        mcp_tool_spec = LlamaIndexMcpToolSpec(
            client=mcp_client,
            allowed_tools=self.mcp_tool.tools,
        )

        self.tools = await mcp_tool_spec.to_tool_list_async()
