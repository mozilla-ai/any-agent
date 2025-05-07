from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

from pydantic import PrivateAttr

from any_agent.config import AgentFramework, MCPSse, MCPStdio
from any_agent.tools.mcp.mcp_connection import _MCPConnection
from any_agent.tools.mcp.mcp_server import _MCPServerBase

if TYPE_CHECKING:
    from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse
    from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
    from agents.mcp.server import MCPServer

mcp_available = False
with suppress(ImportError):
    from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse
    from agents.mcp import (
        MCPServerSseParams as OpenAIInternalMCPServerSseParams,
    )
    from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
    from agents.mcp import (
        MCPServerStdioParams as OpenAIInternalMCPServerStdioParams,
    )
    from mcp.types import Tool as MCPTool  # noqa: TC002

    mcp_available = True


class OpenAIMCPConnection(_MCPConnection["MCPTool"], ABC):
    """Base class for OpenAI MCP connections."""

    _server: "OpenAIInternalMCPServerStdio | OpenAIInternalMCPServerSse | None" = (
        PrivateAttr(default=None)
    )

    @abstractmethod
    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        if not self._server:
            msg = "MCP server is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        await self._exit_stack.enter_async_context(self._server)
        tools = await self._server.list_tools()
        return self._filter_tools(tools)  # type: ignore[return-value]

    @property
    def server(self) -> "MCPServer | None":
        """Return the MCP server instance."""
        return self._server


class OpenAIMCPStdioConnection(OpenAIMCPConnection):
    mcp_tool: MCPStdio

    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        params = OpenAIInternalMCPServerStdioParams(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
        )

        self._server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params=params,
        )
        return await super().list_tools()


class OpenAIMCPSseConnection(OpenAIMCPConnection):
    mcp_tool: MCPSse

    async def list_tools(self) -> list["MCPTool"]:
        """List tools from the MCP server."""
        params = OpenAIInternalMCPServerSseParams(url=self.mcp_tool.url)

        self._server = OpenAIInternalMCPServerSse(
            name="OpenAI MCP Server", params=params
        )

        return await super().list_tools()


class OpenAIMCPServerBase(_MCPServerBase["MCPTool"], ABC):
    framework: Literal[AgentFramework.OPENAI] = AgentFramework.OPENAI

    def _check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,openai]"
        self.mcp_available = mcp_available
        super()._check_dependencies()


class OpenAIMCPServerStdio(OpenAIMCPServerBase):
    mcp_tool: MCPStdio

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["MCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or OpenAIMCPStdioConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


class OpenAIMCPServerSse(OpenAIMCPServerBase):
    mcp_tool: MCPSse

    async def _setup_tools(
        self, mcp_connection: _MCPConnection["MCPTool"] | None = None
    ) -> None:
        mcp_connection = mcp_connection or OpenAIMCPSseConnection(
            mcp_tool=self.mcp_tool
        )
        await super()._setup_tools(mcp_connection)


OpenAIMCPServer = OpenAIMCPServerStdio | OpenAIMCPServerSse
