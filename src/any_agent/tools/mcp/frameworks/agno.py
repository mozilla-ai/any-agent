import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, suppress
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from any_agent.config import (
    AgentFramework,
    MCPParams,
    MCPSseParams,
    MCPStdioParams,
    Tool,
)
from any_agent.tools.mcp.mcp_connection import MCPConnection
from any_agent.tools.mcp.mcp_server import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from agno.tools.mcp import MCPTools as AgnoMCPTools
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    mcp_available = True


class AgnoMCPConnection(BaseModel, ABC):
    mcp_tool: MCPParams
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)

    @abstractmethod
    async def list_tools(self) -> list[Tool]: ...


class AgnoMCPStdioConnection(AgnoMCPConnection):
    mcp_tool: MCPStdioParams

    async def list_tools(self) -> list[Tool]:
        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        server = AgnoMCPTools(
            command=server_params,
            include_tools=list(self.mcp_tool.tools or []),
            env={**os.environ},
        )
        return [await self._exit_stack.enter_async_context(server)]


class AgnoMCPSseConnection(AgnoMCPConnection):
    mcp_tool: MCPSseParams

    async def list_tools(self) -> list[Tool]:
        client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        sse_transport = await self._exit_stack.enter_async_context(client)
        stdio, write = sse_transport
        client_session = ClientSession(stdio, write)
        session = await self._exit_stack.enter_async_context(client_session)
        await session.initialize()
        server = AgnoMCPTools(
            session=session,
            include_tools=list(self.mcp_tool.tools or []),
        )
        return [await self._exit_stack.enter_async_context(server)]


class AgnoMCPServerBase(MCPServerBase, ABC):
    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,agno]"
        self.mcp_available = mcp_available
        super()._check_dependencies()


class AgnoMCPServerStdio(AgnoMCPServerBase):
    mcp_tool: MCPStdioParams

    async def _setup_tools(self, mcp_connection: MCPConnection | None = None) -> None:
        mcp_connection = mcp_connection or AgnoMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class AgnoMCPServerSse(AgnoMCPServerBase):
    mcp_tool: MCPSseParams

    async def _setup_tools(self, mcp_connection: MCPConnection | None = None) -> None:
        mcp_connection = mcp_connection or AgnoMCPSseConnection(mcp_tool=self.mcp_tool)
        await super()._setup_tools(mcp_connection)


AgnoMCPServer = AgnoMCPServerStdio | AgnoMCPServerSse
