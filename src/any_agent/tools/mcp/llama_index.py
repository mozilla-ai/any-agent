"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os

from any_agent.config import MCPStdioParams

from .mcp_server_base import MCPServerBase


class LlamaIndexMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
        from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

        if not isinstance(self.mcp_tool, MCPStdioParams):
            raise NotImplementedError

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
