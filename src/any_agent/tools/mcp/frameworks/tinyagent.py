"""MCP adapter for Tiny framework."""

import os
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta
from typing import Any, Literal

from mcp import Tool

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.mcp_server import MCPServerBase

# Check for MCP dependencies
mcp_available = False
with suppress(ImportError):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    mcp_available = True


class TinyAgentMCPServerBase(MCPServerBase):
    """MCP adapter for Tiny framework."""

    client: Any | None = None
    framework: Literal[AgentFramework.TINYAGENT] = AgentFramework.TINYAGENT
    libraries: str = "any-agent[mcp]"
    session: Any | None = None

    def _check_dependencies(self) -> None:
        """Check if the required libraries are installed."""
        self.mcp_available = mcp_available
        if not self.mcp_available:
            super()._check_dependencies()

    async def _setup_tools(self) -> None:
        """Set up the MCP tools for TinyAgent."""
        if not self.client:
            msg = "MCP client is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        # Setup the client connection using exit stack to manage lifecycle
        stdio, write = await self._exit_stack.enter_async_context(self.client)

        # Create a client session
        client_session = ClientSession(
            stdio,
            write,
            timedelta(seconds=self.mcp_tool.client_session_timeout_seconds)
            if self.mcp_tool.client_session_timeout_seconds
            else None,
        )

        # Start the session
        self.session: ClientSession = await self._exit_stack.enter_async_context(
            client_session
        )
        if not self.session:
            msg = "Failed to create MCP session"
            raise ValueError(msg)
        await self.session.initialize()

        # Get the available tools from the MCP server using schema
        available_tools = await self.session.list_tools()

        # Filter tools if specific tools were requested
        filtered_tools = self._filter_tools(available_tools.tools)

        # Create callable tool functions
        tool_list = []
        for tool_info in filtered_tools:
            tool_list.append(self._create_tool_from_info(tool_info))

        # Store tools as a list
        self.tools = tool_list

    def _create_tool_from_info(self, tool: Tool) -> Callable[..., Any]:
        """Create a tool function from tool information."""
        tool_name = tool.name
        tool_description = tool.description
        input_schema = tool.inputSchema
        session = self.session
        if not self.session:
            msg = "Not connected to MCP server"
            raise ValueError(msg)

        async def tool_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            """Tool function that calls the MCP server."""
            # Combine args and kwargs
            combined_args = {}
            if args and len(args) > 0:
                combined_args = args[0]
            combined_args.update(kwargs)

            if not session:
                msg = "Not connected to MCP server"
                raise ValueError(msg)
            # Call the tool on the MCP server
            try:
                return await session.call_tool(tool_name, combined_args)
            except Exception as e:
                return f"Error calling tool {tool_name}: {e!s}"

        # Set attributes for the tool function
        tool_function.__name__ = tool_name
        tool_function.__doc__ = tool_description
        # this isn't a defined attribute of a callable, but we pass it to tinyagent so that it can use it appropriately
        # when constructing the ToolExecutor.
        tool_function.__input_schema__ = input_schema  # type: ignore[attr-defined]

        return tool_function


class TinyAgentMCPServerStdio(TinyAgentMCPServerBase):
    """MCP adapter for Tiny framework using stdio communication."""

    mcp_tool: MCPStdioParams

    async def _setup_tools(self) -> None:
        server_params = StdioServerParameters(
            command=self.mcp_tool.command,
            args=list(self.mcp_tool.args),
            env={**os.environ},
        )

        self.client = stdio_client(server_params)

        await super()._setup_tools()


class TinyAgentMCPServerSse(TinyAgentMCPServerBase):
    """MCP adapter for Tiny framework using SSE communication."""

    mcp_tool: MCPSseParams

    async def _setup_tools(self) -> None:
        self.client = sse_client(
            url=self.mcp_tool.url,
            headers=dict(self.mcp_tool.headers or {}),
        )

        await super()._setup_tools()


# Union type for Tiny MCP server implementations
TinyAgentMCPServer = TinyAgentMCPServerStdio | TinyAgentMCPServerSse
