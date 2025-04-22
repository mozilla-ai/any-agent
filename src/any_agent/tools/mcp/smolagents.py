"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress
from textwrap import dedent

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams
from any_agent.logging import logger

from .mcp_server_base import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from mcp import StdioServerParameters
    from smolagents.mcp_client import MCPClient, Tool

    mcp_available = True


class SmolagentsMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool, "any-agent[mcp,smolagents]", mcp_available)
        self.exit_stack = AsyncExitStack()
        self.smolagent_tools: list[Tool] | None = None

    async def setup_stdio_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPStdioParams):
            msg = "MCP tool parameters must be of type MCPStdioParams for stdio server."
            raise ValueError(msg)

        server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.smolagent_tools = self.exit_stack.enter_context(
            MCPClient(server_parameters)
        )

    async def setup_sse_tools(self) -> None:
        if not isinstance(self.mcp_tool, MCPSseParams):
            msg = "MCP tool parameters must be of type MCPSseParams for SSE server."
            raise ValueError(msg)

        server_parameters = {
            "url": self.mcp_tool.url,
        }
        self.smolagent_tools = self.exit_stack.enter_context(
            MCPClient(server_parameters)
        )

    async def setup_tools(self) -> None:
        await super().setup_tools()

        if not self.smolagent_tools:
            msg = "Tool collection is not set up. Please call setup_stdio_tools or setup_sse_tools first."
            raise ValueError(msg)

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        if not requested_tools:
            logger.info(
                "No specific tools requested for MCP server, using all available tools:",
            )
            logger.info("Tools available: %s", self.smolagent_tools)
            self.tools = self.smolagent_tools
            return

        filtered_tools = [
            tool for tool in self.smolagent_tools if tool.name in requested_tools
        ]
        if len(filtered_tools) != len(requested_tools):
            tool_names = [tool.name for tool in filtered_tools]
            raise ValueError(
                dedent(f"""Could not find all requested tools in the MCP server:
                            Requested: {requested_tools}
                            Set:   {tool_names}"""),
            )

        self.tools = filtered_tools
