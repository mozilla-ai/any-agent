from abc import ABC, abstractmethod
from collections.abc import Sequence
from textwrap import dedent
from typing import Any

from agents.mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from any_agent.config import AgentFramework, MCPParams, Tool

from .mcp_connection import MCPConnection


class MCPServerBase(BaseModel, ABC):
    mcp_tool: MCPParams
    framework: AgentFramework
    mcp_available: bool = False
    libraries: str = ""

    tools: Sequence[Tool] = Field(default_factory=list)
    tool_names: Sequence[str] = Field(default_factory=list)
    mcp_connection: MCPConnection | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:  # noqa: D102
        self._check_dependencies()

    @abstractmethod
    async def _setup_tools(self, mcp_connection: MCPConnection | None = None) -> None:
        if not mcp_connection:
            msg = "MCP server is not set up. Please call `_setup_tools` from a concrete class."
            raise ValueError(msg)

        self.mcp_connection = mcp_connection
        tools = await mcp_connection.list_tools()
        self.tools = self._filter_tools(tools)

    @property
    def server(self) -> MCPServer:
        """Return the MCP server instance."""
        if not self.mcp_connection or not self.mcp_connection.server:
            msg = "MCP server is not set up. Please call `_setup_tools` from a concrete class."
            raise ValueError(msg)

        return self.mcp_connection.server

    @abstractmethod
    def _check_dependencies(self) -> None:
        if self.mcp_available:
            return

        msg = f"You need to `pip install '{self.libraries}'` to use MCP."
        raise ImportError(msg)

    def _filter_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = list(self.mcp_tool.tools or [])

        if not requested_tools:
            return tools

        found_tools = [tool for tool in tools if tool in requested_tools]

        if len(found_tools) != len(requested_tools):
            error_message = (
                dedent(
                    f"""Could not find all requested tools in the MCP server:
                    Requested: {requested_tools}
                    Set:   {tools}
                """
                ),
            )
            raise ValueError(error_message)
        return tools
