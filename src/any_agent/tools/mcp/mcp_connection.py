"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from any_agent.config import AgentFramework, MCPParams


class MCPConnection(BaseModel, ABC):
    mcp_tool: MCPParams
    framework: AgentFramework
    mcp_available: bool = False
    libraries: str = ""

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def check_dependencies(self) -> None:
        if self.mcp_available:
            return

        msg = f"You need to `pip install '{self.libraries}'` to use MCP."
        raise ImportError(msg)
