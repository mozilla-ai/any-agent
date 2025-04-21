"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress
from textwrap import dedent

from any_agent.config import MCPParams, MCPSseParams, MCPStdioParams
from any_agent.logging import logger

# from any_agent.tools.mcp.mcp_server import MCPServer

with suppress(ImportError):
    from mcp import StdioServerParameters
    from smolagents import ToolCollection


class SmolagentsMCPServer(object):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.exit_stack = AsyncExitStack()
        self.tool_collection: ToolCollection | None = None

    async def setup_stdio_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPStdioParams)

        server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.tool_collection = self.exit_stack.enter_context(
            ToolCollection.from_mcp(server_parameters, trust_remote_code=True)
        )

    async def setup_sse_tools(self) -> None:
        assert isinstance(self.mcp_tool, MCPSseParams)

        server_parameters = {
            "url": self.mcp_tool.url,
        }
        self.tool_collection = self.exit_stack.enter_context(
            ToolCollection.from_mcp(server_parameters, trust_remote_code=True)
        )

    async def setup_tools(self) -> None:
        await super().setup_tools()

        assert self.tool_collection
        # Store the context manager itself

        tools = self.tool_collection.tools

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        if not requested_tools:
            logger.info(
                "No specific tools requested for MCP server, using all available tools:",
            )
            logger.info(f"Tools available: {tools}")
            self.tools = tools
            return

        filtered_tools = [tool for tool in tools if tool.name in requested_tools]
        if len(filtered_tools) != len(requested_tools):
            tool_names = [tool.name for tool in filtered_tools]
            raise ValueError(
                dedent(f"""Could not find all requested tools in the MCP server:
                            Requested: {requested_tools}
                            Set:   {tool_names}"""),
            )

        self.tools = filtered_tools
