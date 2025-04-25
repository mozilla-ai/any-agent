import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
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

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,smolagents]"
        self.mcp_available = mcp_available
        super()._check_dependencies()

    @abstractmethod
    async def _setup_tools(self) -> None:
        """Set up the Smolagents MCP server with the provided configuration."""
        if not self.smolagent_tools:
            msg = "Tool collection is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        self.tools = self._filter_tools(self.smolagent_tools)


class SmolagentsMCPServerStdio(SmolagentsMCPServerBase):
    mcp_tool: MCPStdioParams

    async def _setup_tools(self) -> None:
        server_parameters = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.smolagent_tools = self._exit_stack.enter_context(
            MCPClient(server_parameters)
        )

        await super()._setup_tools()


class SmolagentsMCPServerSse(SmolagentsMCPServerBase):
    mcp_tool: MCPSseParams

    async def _setup_tools(self) -> None:
        server_parameters = {
            "url": self.mcp_tool.url,
        }
        self.smolagent_tools = self._exit_stack.enter_context(
            MCPClient(server_parameters)
        )

        await super()._setup_tools()


SmolagentsMCPServer = SmolagentsMCPServerStdio | SmolagentsMCPServerSse
