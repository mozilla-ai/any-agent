import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Literal

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.mcp_server import MCPServerBase

mcp_available = False
with suppress(ImportError):
    from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
    from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

    mcp_available = True


class LlamaIndexMCPServerBase(MCPServerBase, ABC):
    client: LlamaIndexMCPClient | None = None
    framework: Literal[AgentFramework.LLAMA_INDEX] = AgentFramework.LLAMA_INDEX

    def check_dependencies(self) -> None:
        """Check if the required dependencies for the MCP server are available."""
        self.libraries = "any-agent[mcp,llama_index]"
        self.mcp_available = mcp_available
        super().check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up the LlamaIndex MCP server with the provided configuration."""
        if not self.client:
            msg = "MCP client is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        mcp_tool_spec = LlamaIndexMcpToolSpec(
            client=self.client,
            allowed_tools=list(self.mcp_tool.tools or []),
        )

        self.tools = await mcp_tool_spec.to_tool_list_async()


class LlamaIndexMCPServerStdio(LlamaIndexMCPServerBase):
    mcp_tool: MCPStdioParams

    async def setup_tools(self) -> None:
        self.client = LlamaIndexMCPClient(
            command_or_url=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

        await super().setup_tools()


class LlamaIndexMCPServerSse(LlamaIndexMCPServerBase):
    mcp_tool: MCPSseParams

    async def setup_tools(self) -> None:
        self.client = LlamaIndexMCPClient(command_or_url=self.mcp_tool.url)

        await super().setup_tools()


LlamaIndexMCPServer = LlamaIndexMCPServerStdio | LlamaIndexMCPServerSse
