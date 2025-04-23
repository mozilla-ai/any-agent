from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from any_agent.config import AgentFramework, MCPParams, Tool


class MCPServerBase(BaseModel, ABC):
    mcp_tool: MCPParams
    framework: AgentFramework
    mcp_available: bool = False
    libraries: str = ""

    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)
    tools: Sequence[Tool] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        self.check_dependencies()

    @abstractmethod
    async def setup_tools(self) -> None: ...

    @abstractmethod
    def check_dependencies(self) -> None:
        if self.mcp_available:
            return

        msg = f"You need to `pip install '{self.libraries}'` to use MCP."
        raise ImportError(msg)
