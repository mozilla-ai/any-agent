import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.mcp_server import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from agno.tools.mcp import MCPTools as AgnoMCPTools
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    mcp_available = True


class AgnoMCPServerBase(MCPServerBase, ABC):
    server: AgnoMCPTools | None = None
    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO

    def check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,agno]"
        self.mcp_available = mcp_available
        super().check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up the Agno MCP server with the provided configuration."""
        if not self.server:
            msg = "MCP server is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        self.tools = [await self._exit_stack.enter_async_context(self.server)]  # type: ignore[arg-type]


class AgnoMCPServerStdio(AgnoMCPServerBase):
    mcp_tool: MCPStdioParams

    async def setup_tools(self) -> None:
        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self.server = AgnoMCPTools(
            command=server_params,
            include_tools=list(self.mcp_tool.tools or []),
            env={**os.environ},
        )

        await super().setup_tools()


class AgnoMCPServerSse(AgnoMCPServerBase):
    mcp_tool: MCPSseParams

    async def setup_tools(self) -> None:
        client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        sse_transport = await self._exit_stack.enter_async_context(client)
        stdio, write = sse_transport
        client_session = ClientSession(stdio, write)
        session = await self._exit_stack.enter_async_context(client_session)
        await session.initialize()
        self.server = AgnoMCPTools(
            session=session,
            include_tools=list(self.mcp_tool.tools or []),
        )

        await super().setup_tools()


AgnoMCPServer = AgnoMCPServerStdio | AgnoMCPServerSse
