import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.mcp_server import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        SseServerParams as GoogleSseServerParameters,
    )
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        StdioServerParameters as GoogleStdioServerParameters,
    )

    mcp_available = True


class GoogleMCPServerBase(MCPServerBase, ABC):
    server: GoogleMCPToolset | None = None
    framework: Literal[AgentFramework.GOOGLE] = AgentFramework.GOOGLE

    def check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,google]"
        self.mcp_available = mcp_available
        super().check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        if not self.server:
            msg = "MCP server is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        await self._exit_stack.enter_async_context(self.server)
        self.tools = await self.server.load_tools()


class GoogleMCPServerStdio(GoogleMCPServerBase):
    mcp_tool: MCPStdioParams

    async def setup_tools(self) -> None:
        params = GoogleStdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        self.server = GoogleMCPToolset(connection_params=params)

        await super().setup_tools()


class GoogleMCPServerSse(GoogleMCPServerBase):
    mcp_tool: MCPSseParams

    async def setup_tools(self) -> None:
        params = GoogleSseServerParameters(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        self.server = GoogleMCPToolset(connection_params=params)

        await super().setup_tools()


GoogleMCPServer = GoogleMCPServerStdio | GoogleMCPServerSse
