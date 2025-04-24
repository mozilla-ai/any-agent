from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.logging import logger
from any_agent.tools.mcp.mcp_server import MCPServerBase

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

    mcp_available = True


class OpenAIMCPServerBase(MCPServerBase, ABC):
    server: OpenAIInternalMCPServerStdio | OpenAIInternalMCPServerSse | None = None
    framework: Literal[AgentFramework.OPENAI] = AgentFramework.OPENAI

    def check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,openai]"
        self.mcp_available = mcp_available
        super().check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up the OpenAI MCP server with the provided configuration."""
        if not self.server:
            msg = "MCP server is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        await self._exit_stack.enter_async_context(self.server)
        # Get tools from the server
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools",
        )
        self.tools = await self.server.list_tools()  # type: ignore[assignment]


class OpenAIMCPServerStdio(OpenAIMCPServerBase):
    mcp_tool: MCPStdioParams

    async def setup_tools(self) -> None:
        params = OpenAIInternalMCPServerStdioParams(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
        )

        self.server = OpenAIInternalMCPServerStdio(
            name="OpenAI MCP Server",
            params=params,
        )

        await super().setup_tools()


class OpenAIMCPServerSse(OpenAIMCPServerBase):
    mcp_tool: MCPSseParams

    async def setup_tools(self) -> None:
        params = OpenAIInternalMCPServerSseParams(url=self.mcp_tool.url)

        self.server = OpenAIInternalMCPServerSse(
            name="OpenAI MCP Server", params=params
        )

        await super().setup_tools()


OpenAIMCPServer = OpenAIMCPServerStdio | OpenAIMCPServerSse
