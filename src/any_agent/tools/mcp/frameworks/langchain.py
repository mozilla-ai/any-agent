from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import timedelta
from typing import Any, Literal, Union

from pydantic import Field, PrivateAttr

from any_agent.config import AgentFramework, MCPSse, MCPStdio, MCPStreamableHttp
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

mcp_available = False
with suppress(ImportError):
    from langchain_core.tools import BaseTool  # noqa: TC002
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    mcp_available = True


class LangchainMCPConnection(_MCPConnection["BaseTool"], ABC):
    """Base class for LangChain MCP connections."""

    _client: Any | None = PrivateAttr(default=None)
    session: Union["ClientSession", None] = None

    @abstractmethod
    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        if not self._client:
            msg = "MCP client is not set up. Please call `list_tools` from a concrete class."
            raise ValueError(msg)

        tools = await load_mcp_tools(self.session)
        return self._filter_tools(tools)  # type: ignore[return-value]


class LangchainMCPStdioConnection(LangchainMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env=self.mcp_tool.env,
        )

        self._client = stdio_client(server_params)

        read, write = await self._exit_stack.enter_async_context(self._client)
        kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            kwargs["read_timeout_seconds"] = timedelta(
                seconds=self.mcp_tool.client_session_timeout_seconds
            )
        client_session = ClientSession(
            read,
            write,
            **kwargs,  # type: ignore[arg-type]
        )
        self.session = await self._exit_stack.enter_async_context(client_session)

        await self.session.initialize()

        return await super().list_tools()


class LangchainMCPSseConnection(LangchainMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        sse_kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            sse_kwargs["sse_read_timeout"] = (
                self.mcp_tool.client_session_timeout_seconds
            )
        self._client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
            **sse_kwargs,  # type: ignore[arg-type]
        )
        read, write = await self._exit_stack.enter_async_context(self._client)
        session_kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            session_kwargs["read_timeout_seconds"] = timedelta(
                seconds=self.mcp_tool.client_session_timeout_seconds
            )
        client_session = ClientSession(
            read,
            write,
            **session_kwargs,  # type: ignore[arg-type]
        )
        self.session = await self._exit_stack.enter_async_context(client_session)

        await self.session.initialize()

        return await super().list_tools()


class LangchainMCPStreamableHttpConnection(LangchainMCPConnection):
    mcp_tool: MCPStreamableHttp
    _get_session_id_callback: Callable[[], str | None] | None = None

    def get_session_id(self) -> str | None:
        """Retrieve session ID, if it has been established."""
        if self._get_session_id_callback:
            return self._get_session_id_callback()
        return None

    async def list_tools(self) -> list["BaseTool"]:
        """List tools from the MCP server."""
        http_kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            http_kwargs["sse_read_timeout"] = timedelta(
                seconds=self.mcp_tool.client_session_timeout_seconds
            )
        self._client = streamablehttp_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
            **http_kwargs,  # type: ignore[arg-type]
        )
        streamablehttp_transport = await self._exit_stack.enter_async_context(
            self._client
        )
        read_stream, write_stream, get_session_id_callback = streamablehttp_transport
        self._get_session_id_callback = get_session_id_callback
        session_kwargs = {}
        if self.mcp_tool.client_session_timeout_seconds:
            session_kwargs["read_timeout_seconds"] = timedelta(
                seconds=self.mcp_tool.client_session_timeout_seconds
            )
        client_session = ClientSession(read_stream, write_stream, **session_kwargs)  # type: ignore[arg-type]
        self.session = await self._exit_stack.enter_async_context(client_session)
        await self.session.initialize()
        return await super().list_tools()


class LangchainMCPServerBase(_MCPServerBase["BaseTool"], ABC):
    framework: Literal[AgentFramework.LANGCHAIN] = AgentFramework.LANGCHAIN
    tools: Sequence["BaseTool"] = Field(default_factory=list)

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


class LangchainMCPServerStreamableHttp(LangchainMCPServerBase):
    mcp_tool: MCPStreamableHttp

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["BaseTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or LangchainMCPStreamableHttpConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


LangchainMCPServer = (
    LangchainMCPServerStdio | LangchainMCPServerSse | LangchainMCPServerStreamableHttp
)
