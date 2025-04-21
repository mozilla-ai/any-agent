"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from contextlib import AsyncExitStack, suppress
from typing import Literal


from pydantic import ConfigDict

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams, Tool

from any_agent.tools.mcp.mcp_connection import MCPConnection


with suppress(ImportError):
    from agno.tools.mcp import MCPTools as AgnoMCPTools
    from mcp import ClientSession
    from mcp.client.sse import sse_client


class AgnoMCPToolConnectionBase(MCPConnection):
    server: AgnoMCPTools | None = None
    exit_stack: AsyncExitStack = AsyncExitStack()
    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO
    

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def setup(self) -> list[Tool]:
        """Set up the Agno MCP server with the provided configuration."""
        assert self.server
        return [await self.exit_stack.enter_async_context(self.server)]

class AgnoMCPToolConnectionStdio(AgnoMCPToolConnectionBase):
    mcp_tool: MCPStdioParams
    
    async def setup(self) -> list[Tool]:
        server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
        self.server = AgnoMCPTools(
            command=server_params,
            include_tools=self.mcp_tool.tools,
            env={**os.environ},
        )

        return await super().setup()

class AgnoMCPToolConnectionSse(AgnoMCPToolConnectionBase):
    mcp_tool: MCPSseParams

    async def setup(self) -> list[Tool]:
        client = sse_client(
            url=self.mcp_tool.url,
            headers=self.mcp_tool.headers,
        )
        sse_transport = await self.exit_stack.enter_async_context(client)
        stdio, write = sse_transport
        client_session = ClientSession(stdio, write)
        session = await self.exit_stack.enter_async_context(client_session)
        await session.initialize()
        self.server = AgnoMCPTools(
            session=session,
            include_tools=self.mcp_tool.tools,
        )

        return await super().setup()


AgnoMCPToolConnection = AgnoMCPToolConnectionStdio | AgnoMCPToolConnectionSse
