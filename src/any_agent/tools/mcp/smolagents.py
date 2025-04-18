"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import suppress
from textwrap import dedent

from any_agent.config import MCPParams, MCPStdioParams
from any_agent.logging import logger

from .mcp_server_base import MCPServerBase

with suppress(ImportError):
    from mcp import StdioServerParameters


class SmolagentsMCPServerStdio(MCPServerBase):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.context = None
        self.tool_collection = None

    async def setup_tools(self) -> None:
        from smolagents import ToolCollection

        if not isinstance(self.mcp_tool, MCPStdioParams):
            raise NotImplementedError

        self.server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

        # Store the context manager itself
        self.context = ToolCollection.from_mcp(
            self.server_parameters, trust_remote_code=True
        )
        # Enter the context
        self.tool_collection = self.context.__enter__()  # type: ignore[attr-defined]
        tools = self.tool_collection.tools  # type: ignore[attr-defined]

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        if requested_tools:
            filtered_tools = [tool for tool in tools if tool.name in requested_tools]
            if len(filtered_tools) != len(requested_tools):
                tool_names = [tool.name for tool in filtered_tools]
                raise ValueError(
                    dedent(f"""Could not find all requested tools in the MCP server:
                                Requested: {requested_tools}
                                Set:   {tool_names}""")
                )
            self.tools = filtered_tools
        else:
            logger.info(
                "No specific tools requested for MCP server, using all available tools:"
            )
            logger.info(f"Tools available: {tools}")
            self.tools = tools
