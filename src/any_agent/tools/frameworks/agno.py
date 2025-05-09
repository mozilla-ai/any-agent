from collections.abc import Callable
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

AgnoToolType = Callable[..., Any]


class AgnoTool(AnyToolBase[AgnoToolType]):
    """Wrapper class for the Tools used by Agno."""

    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> AgnoToolType:
        # Agno lets you pass callables directly in as tools ❤️
        return tool

    @property
    def name(self) -> str:
        """Returns the name of the tool."""
        return self.tool.__name__
