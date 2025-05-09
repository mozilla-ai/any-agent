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
    from agno.tools.mcp import MCPTools as AgnoMCPTools
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    mcp_available = True


class AgnoMCPConnection(_MCPConnection["AgnoMCPTools"], ABC):
    _server: "AgnoMCPTools | None" = PrivateAttr(default=None)

    @abstractmethod
    async def list_tools(self) -> list["AgnoMCPTools"]:
        """List tools from the MCP server."""
        if self._server is None:
            msg = "MCP server is not set up. Please call `list_tools` from a concrete class."
            raise ValueError(msg)

        tools = await self._exit_stack.enter_async_context(self._server)
        return [tools]


class AgnoMCPStdioConnection(AgnoMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["AgnoMCPTools"]:
        """List tools from the MCP server."""
        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self._server = AgnoMCPTools(
            command=server_params,
            include_tools=list(self.mcp_tool.tools or []),
            env={**os.environ},
        )
        return await super().list_tools()


class AgnoMCPSseConnection(AgnoMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["AgnoMCPTools"]:
        """List tools from the MCP server."""
        client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        sse_transport = await self._exit_stack.enter_async_context(client)
        stdio, write = sse_transport
        client_session = ClientSession(stdio, write)
        session = await self._exit_stack.enter_async_context(client_session)
        await session.initialize()
        self._server = AgnoMCPTools(
            session=session,
            include_tools=list(self.mcp_tool.tools or []),
        )
        return await super().list_tools()


class AgnoMCPServerBase(_MCPServerBase["AgnoMCPTools"], ABC):
    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,agno]"
        self.mcp_available = mcp_available
        super()._check_dependencies()


class AgnoMCPServerStdio(AgnoMCPServerBase):
    mcp_tool: MCPStdio

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["AgnoMCPTools"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or AgnoMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class AgnoMCPServerSse(AgnoMCPServerBase):
    mcp_tool: MCPSse

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["AgnoMCPTools"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or AgnoMCPSseConnection(mcp_tool=self.mcp_tool)
        await super()._setup_tools(mcp_connection)


AgnoMCPServer = AgnoMCPServerStdio | AgnoMCPServerSse
