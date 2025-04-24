import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from textwrap import dedent
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.logging import logger
from any_agent.tools.mcp.mcp_server import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from mcp import StdioServerParameters
    from smolagents.mcp_client import MCPClient
    from smolagents.tools import Tool as SmolagentsTool

    mcp_available = True


class SmolagentsMCPServerBase(MCPServerBase, ABC):
    smolagent_tools: Sequence[SmolagentsTool] | None = None
    framework: Literal[AgentFramework.SMOLAGENTS] = AgentFramework.SMOLAGENTS

    def check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,smolagents]"
        self.mcp_available = mcp_available
        super().check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up the Smolagents MCP server with the provided configuration."""
        if not self.smolagent_tools:
            msg = "Tool collection is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = list(self.mcp_tool.tools or [])
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
        if len(filtered_tools) == len(requested_tools):
            self.tools = filtered_tools
            return

        tool_names = [tool.name for tool in filtered_tools]
        raise ValueError(
            dedent(f"""Could not find all requested tools in the MCP server:
                        Requested: {requested_tools}
                        Set:   {tool_names}"""),
        )


class SmolagentsMCPServerStdio(SmolagentsMCPServerBase):
    mcp_tool: MCPStdioParams

    async def setup_tools(self) -> None:
        server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.smolagent_tools = self._exit_stack.enter_context(
            MCPClient(server_parameters)
        )

        await super().setup_tools()


class SmolagentsMCPServerSse(SmolagentsMCPServerBase):
    mcp_tool: MCPSseParams

    async def setup_tools(self) -> None:
        server_parameters = {
            "url": self.mcp_tool.url,
        }
        self.smolagent_tools = self._exit_stack.enter_context(
            MCPClient(server_parameters)
        )

        await super().setup_tools()


SmolagentsMCPServer = SmolagentsMCPServerStdio | SmolagentsMCPServerSse
