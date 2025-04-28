from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from any_agent.config import Tool


class FakeMCPConnection(BaseModel):
    tools: Sequence[Tool] = Field(default_factory=list)

    async def list_tools(self) -> list[Tool]:
        return list(self.tools)


class MCPConnection(Protocol):
    async def list_tools(self) -> list[Tool]: ...
