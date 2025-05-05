from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AsyncExitStack
from textwrap import dedent
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from any_agent.config import MCPParams, Tool

if TYPE_CHECKING:
    from agents.mcp.server import MCPServer


@runtime_checkable
class HasName(Protocol):
    """Protocol for objects that have a name."""

    name: str


ToolsWithName = Sequence[str] | Sequence[HasName]


class MCPConnection(BaseModel, ABC):
    mcp_tool: MCPParams
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def list_tools(self) -> list[Tool]:
        """List tools from the MCP server."""

    @property
    def server(self) -> "MCPServer | None":
        """Return the MCP server instance."""
        return None

    def _filter_tools(self, tools: ToolsWithName) -> ToolsWithName:
        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = list(self.mcp_tool.tools or [])

        if not requested_tools:
            return tools

        tool_names = [
            tool.name if isinstance(tool, HasName) else tool for tool in tools
        ]

        found_tools = [tool for tool in tool_names if tool in requested_tools]

        if len(found_tools) != len(requested_tools):
            error_message = (
                dedent(
                    f"""Could not find all requested tools in the MCP server:
                    Requested ({len(requested_tools)}): {requested_tools}
                    Set ({len(tool_names)}):   {tool_names}
                """
                ),
            )
            raise ValueError(error_message)
        return tools
