"""Tools for managing MCP (Model Context Protocol) connections and resources."""

import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from any_agent.config import MCPParams, MCPStdioParams
from any_agent.logging import logger

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    mcp_available = True
except ImportError:
    mcp_available = False

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
    from smolagents import ToolCollection


class MCPServerBase(ABC):
    """Base class for MCP tools managers across different frameworks."""

    def __init__(self, mcp_tool: MCPParams):
        if not mcp_available:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP tools."
            raise ImportError(msg)

        # Store the original tool configuration
        self.mcp_tool = mcp_tool

        # Initialize tools list (to be populated by subclasses)
        self.tools: Sequence[Any] = []

    @abstractmethod
    async def setup_tools(self) -> None:
        """Set up tools. To be implemented by subclasses."""


class SmolagentsMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for smolagents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.exit_stack = AsyncExitStack()
        self.tool_collection: ToolCollection | None = None

    async def setup_tools(self) -> None:
        from smolagents import ToolCollection

        if isinstance(self.mcp_tool, MCPStdioParams):
            server_parameters = StdioServerParameters(
                command=self.mcp_tool.command,
                args=self.mcp_tool.args,
                env={**os.environ},
            )
        else:
            server_parameters = {
                "url": self.mcp_tool.url,
            }

        # Store the context manager itself
        self.tool_collection = self.exit_stack.enter_context(
            ToolCollection.from_mcp(server_parameters, trust_remote_code=True)
        )
        tools = self.tool_collection.tools

        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = self.mcp_tool.tools
        if requested_tools:
            filtered_tools = [tool for tool in tools if tool.name in requested_tools]
            if len(filtered_tools) != len(requested_tools):
                tool_names = [tool.name for tool in filtered_tools]
                raise ValueError(
                    dedent(f"""Could not find all requested tools in the MCP server:
                                Requested: {requested_tools}
                                Set:   {tool_names}"""),
                )
            self.tools = filtered_tools
        else:
            logger.info(
                "No specific tools requested for MCP server, using all available tools:",
            )
            logger.info(f"Tools available: {tools}")
            self.tools = tools


class OpenAIMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for OpenAI agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server: OpenAIInternalMCPServerStdio | None = None
        self.exit_stack = AsyncExitStack()

    async def setup_tools(self) -> None:
        """Set up the OpenAI MCP server with the provided configuration."""
        from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse
        from agents.mcp import (
            MCPServerSseParams as OpenAIInternalMCPServerSseParams,
        )
        from agents.mcp import MCPServerStdio as OpenAIInternalMCPServerStdio
        from agents.mcp import (
            MCPServerStdioParams as OpenAIInternalMCPServerStdioParams,
        )

        if isinstance(self.mcp_tool, MCPStdioParams):
            params = OpenAIInternalMCPServerStdioParams(
                command=self.mcp_tool.command,
                args=self.mcp_tool.args,
            )

            self.server = OpenAIInternalMCPServerStdio(
                name="OpenAI MCP Server",
                params=params,
            )
        else:
            params = OpenAIInternalMCPServerSseParams(
                url=self.mcp_tool.url,
            )

            self.server = OpenAIInternalMCPServerSse(
                name="OpenAI MCP Server", params=params
            )

        await self.exit_stack.enter_async_context(self.server)
        # Get tools from the server
        self.tools = await self.server.list_tools()
        logger.warning(
            "OpenAI MCP currently does not support filtering MCP available tools",
        )


class LangchainMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for LangChain agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.client = None
        self.tools = []
        self.session: ClientSession = None
        self.exit_stack = AsyncExitStack()

    async def setup_tools(self) -> None:
        """Set up the LangChain MCP server with the provided configuration."""
        from langchain_mcp_adapters.tools import load_mcp_tools

        if isinstance(self.mcp_tool, MCPStdioParams):
            server_params = StdioServerParameters(
                command=self.mcp_tool.command,
                args=self.mcp_tool.args,
                env={**os.environ},
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
        else:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url=self.mcp_tool.url,
                    headers=self.mcp_tool.headers,
                )
            )
            stdio, write = sse_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()
        # List available tools
        self.tools = await load_mcp_tools(self.session)


class GoogleMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.server: GoogleMCPToolset | None = None
        self.exit_stack = AsyncExitStack()

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
        from google.adk.tools.mcp_tool.mcp_toolset import (
            SseServerParams as GoogleSseServerParameters,
        )
        from google.adk.tools.mcp_tool.mcp_toolset import (
            StdioServerParameters as GoogleStdioServerParameters,
        )

        if isinstance(self.mcp_tool, MCPStdioParams):
            params = GoogleStdioServerParameters(
                command=self.mcp_tool.command,
                args=self.mcp_tool.args,
                env={**os.environ},
            )
        else:
            params = GoogleSseServerParameters(
                url=self.mcp_tool.url,
                headers=self.mcp_tool.headers,
            )

        toolset = GoogleMCPToolset(connection_params=params)
        await self.exit_stack.enter_async_context(toolset)
        tools = await toolset.load_tools()
        self.tools = tools
        self.server = toolset


class LlamaIndexMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Google agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)

    async def setup_tools(self) -> None:
        """Set up the Google MCP server with the provided configuration."""
        from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
        from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

        if isinstance(self.mcp_tool, MCPStdioParams):
            mcp_client = LlamaIndexMCPClient(
                command_or_url=self.mcp_tool.command,
                args=self.mcp_tool.args,
                env={**os.environ},
            )
        else:
            mcp_client = LlamaIndexMCPClient(command_or_url=self.mcp_tool.url)

        mcp_tool_spec = LlamaIndexMcpToolSpec(
            client=mcp_client,
            allowed_tools=self.mcp_tool.tools,
        )

        self.tools = await mcp_tool_spec.to_tool_list_async()


class AgnoMCPServer(MCPServerBase):
    """Implementation of MCP tools manager for Agno agents."""

    def __init__(self, mcp_tool: MCPParams):
        super().__init__(mcp_tool)
        self.exit_stack = AsyncExitStack()

    async def setup_tools(self) -> None:
        """Set up the Agno MCP server with the provided configuration."""
        from agno.tools.mcp import MCPTools as AgnoMCPTools

        if isinstance(self.mcp_tool, MCPStdioParams):
            server_params = f"{self.mcp_tool.command} {' '.join(self.mcp_tool.args)}"
            self.server = AgnoMCPTools(
                command=server_params,
                include_tools=self.mcp_tool.tools,
                env={**os.environ},
            )
        else:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url=self.mcp_tool.url,
                    headers=self.mcp_tool.headers,
                )
            )
            stdio, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()
            self.server = AgnoMCPTools(
                session=session,
                include_tools=self.mcp_tool.tools,
            )
        self.tools = [await self.exit_stack.enter_async_context(self.server)]
