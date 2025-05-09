import os
from abc import ABC, abstractmethod
from contextlib import suppress
from datetime import timedelta
from typing import Any, Literal

from pydantic import PrivateAttr

from any_agent.config import AgentFramework, MCPSse, MCPStdio
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

mcp_available = False
with suppress(ImportError):
    from langchain.tools import BaseTool  # noqa: TC002
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    mcp_available = True


class LangchainMCPConnection(_MCPConnection["BaseTool"], ABC):
    """Base class for LangChain MCP connections."""

    _client: Any | None = PrivateAttr(default=None)

    @abstractmethod
    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        if not self._client:
            msg = "MCP client is not set up. Please call `list_tools` from a concrete class."
            raise ValueError(msg)

        stdio, write = await self._exit_stack.enter_async_context(self._client)

        client_session = ClientSession(
            stdio,
            write,
            timedelta(seconds=self.mcp_tool.client_session_timeout_seconds)
            if self.mcp_tool.client_session_timeout_seconds
            else None,
        )
        session = await self._exit_stack.enter_async_context(client_session)

        await session.initialize()

        tools = await load_mcp_tools(session)
        return self._filter_tools(tools)  # type: ignore[return-value]


class LangchainMCPStdioConnection(LangchainMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

        self._client = stdio_client(server_params)

        return await super().list_tools()


class LangchainMCPSseConnection(LangchainMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        self._client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )
        return await super().list_tools()


class LangchainMCPServerBase(_MCPServerBase["BaseTool"], ABC):
    framework: Literal[AgentFramework.LANGCHAIN] = AgentFramework.LANGCHAIN

    def _check_dependencies(self) -> None:
        self.libraries = "any-agent[mcp,langchain]"
        self.mcp_available = mcp_available
        super()._check_dependencies()


class LangchainMCPServerStdio(LangchainMCPServerBase):
    mcp_tool: MCPStdio

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["BaseTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or LangchainMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class LangchainMCPServerSse(LangchainMCPServerBase):
    mcp_tool: MCPSse

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["BaseTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or LangchainMCPSseConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


LangchainMCPServer = LangchainMCPServerStdio | LangchainMCPServerSse
