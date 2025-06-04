from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from any_agent.config import AgentFramework, MCPParams

from .mcp_connection import _MCPConnection

if TYPE_CHECKING:
    from agents.mcp.server import MCPServer

T_co = TypeVar("T_co", covariant=True)


class _MCPServerBase(BaseModel, ABC, Generic[T_co]):
    mcp_tool: MCPParams
    framework: AgentFramework
    mcp_available: bool = False
    libraries: str = ""

    tools: Sequence[T_co] = Field(default_factory=list)
    tool_names: Sequence[str] = Field(default_factory=list)
    mcp_connection: _MCPConnection[T_co] | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        self._check_dependencies()

    @abstractmethod
    async def _setup_tools(
        self, mcp_connection: _MCPConnection[T_co] | None = None
    ) -> None:
        if not mcp_connection:
            msg = "MCP server is not set up. Please call `_setup_tools` from a concrete class."
            raise ValueError(msg)

        self.mcp_connection = mcp_connection
        self.tools = await mcp_connection.list_tools()

    async def aclose(self) -> None:
        """Close the MCP server and clean up resources."""
        if self.mcp_connection:
            await self.mcp_connection.aclose()
            self.mcp_connection = None
        self.tools = []

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager and clean up resources."""
        await self.aclose()

    @property
    def server(self) -> "MCPServer":
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
