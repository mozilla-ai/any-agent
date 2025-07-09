from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, PrivateAttr

from any_agent.config import (
    AgentFramework,
    MCPSse,
    MCPStdio,
)
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

try:
    from google.adk.tools.mcp_tool import MCPTool as GoogleMCPTool
    from google.adk.tools.mcp_tool import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        SseConnectionParams as GoogleSseServerParameters,
    )
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp import (
        StdioServerParameters as GoogleStdioServerParameters,
    )

    mcp_available = True
except ImportError:
    mcp_available = False


class GoogleMCPConnection(_MCPConnection["GoogleMCPTool"], ABC):
    """Base class for Google MCP connections."""

    _params: "GoogleStdioServerParameters | GoogleSseServerParameters | StdioConnectionParams | None" = PrivateAttr(
        default=None
    )

    @abstractmethod
    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        if not self._params:
            msg = "MCP params is not set up. Please call `list_tools` from a concrete class."
            raise ValueError(msg)

        server = GoogleMCPToolset(connection_params=self._params)
        tools = await server.get_tools()
        return self._filter_tools(tools)  # type: ignore[return-value]


class GoogleMCPStdioConnection(GoogleMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        server_params = GoogleStdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env=self.mcp_tool.env,
        )

        timeout = self.mcp_tool.client_session_timeout_seconds
        if timeout is None:
            self._params = server_params
        else:
            self._params = StdioConnectionParams(
                server_params=server_params,
                timeout=timeout,
            )
        return await super().list_tools()


class GoogleMCPSseConnection(GoogleMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["GoogleMCPTool"]:
        """List tools from the MCP server."""
        timeout = self.mcp_tool.client_session_timeout_seconds
        if timeout is None:
            self._params = GoogleSseServerParameters(
                url=self.mcp_tool.url,
                headers=dict(self.mcp_tool.headers or {}),
            )
        else:
            self._params = GoogleSseServerParameters(
                url=self.mcp_tool.url,
                headers=dict(self.mcp_tool.headers or {}),
                timeout=timeout,
                sse_read_timeout=timeout,
            )
        return await super().list_tools()


class GoogleMCPServerBase(_MCPServerBase["GoogleMCPTool"], ABC):
    framework: Literal[AgentFramework.GOOGLE] = AgentFramework.GOOGLE
    tools: Sequence["GoogleMCPTool"] = Field(default_factory=list)

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
