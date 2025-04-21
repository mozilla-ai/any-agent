"""Tools for managing MCP (Model Context Protocol) connections and resources."""

from abc import ABC, abstractmethod
from enum import Enum, auto

from pydantic import BaseModel

from any_agent.config import MCPParams, AgentFramework


class MCPConnection(BaseModel, ABC):
    mcp_tool: MCPParams
    framework: AgentFramework

    @abstractmethod
    def setup(self) -> None:
        ...

