import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Literal

from pydantic import PrivateAttr

from any_agent.config import (
    AgentFramework,
    MCPSse,
    MCPStdio,
)
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

mcp_available = False
with suppress(ImportError):
    from google.adk.tools.mcp_tool import MCPTool as GoogleMCPTool
    from google.adk.tools.mcp_tool import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        SseServerParams as GoogleSseServerParameters,
    )
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        StdioServerParameters as GoogleStdioServerParameters,
    )

    mcp_available = True


class GoogleMCPConnection(_MCPConnection["GoogleMCPTool"], ABC):
    """Base class for Google MCP connections."""

    _params: "GoogleStdioServerParameters | GoogleSseServerParameters | None" = (
        PrivateAttr(default=None)
    )

    @abstractmethod
    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        if not self._params:
            msg = "MCP params is not set up. Please call `list_tools` from a concrete class."
            raise ValueError(msg)

        server = GoogleMCPToolset(connection_params=self._params)
        await self._exit_stack.enter_async_context(server)
        tools = await server.load_tools()
        return self._filter_tools(tools)  # type: ignore[return-value]


class GoogleMCPStdioConnection(GoogleMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        self._params = GoogleStdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )
        return await super().list_tools()


class GoogleMCPSseConnection(GoogleMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        self._params = GoogleSseServerParameters(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        return await super().list_tools()


class GoogleMCPServerBase(_MCPServerBase["GoogleMCPTool"], ABC):
    framework: Literal[AgentFramework.GOOGLE] = AgentFramework.GOOGLE

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,google]"
        self.mcp_available = mcp_available
        super()._check_dependencies()


class GoogleMCPServerStdio(GoogleMCPServerBase):
    mcp_tool: MCPStdio

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["GoogleMCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or GoogleMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class GoogleMCPServerSse(GoogleMCPServerBase):
    mcp_tool: MCPSse

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["GoogleMCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or GoogleMCPSseConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


GoogleMCPServer = GoogleMCPServerStdio | GoogleMCPServerSse
