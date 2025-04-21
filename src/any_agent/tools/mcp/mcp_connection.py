"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from enum import Enum, auto

from pydantic import BaseModel, Field

from any_agent.config import MCPParams


class MCPToolType(str, Enum):
    STDIO = auto()
    SSE = auto()


class MCPConnection(BaseModel, ABC):
    mcp_tool: MCPParams
    type_: MCPToolType = Field(alias="type")

    @abstractmethod
    def setup(self) -> None:
        ...

