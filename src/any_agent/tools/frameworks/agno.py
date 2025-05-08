from typing import Any, Literal

from any_agent.config import AgentFramework, Tool
from any_agent.tools.any_tool import AnyToolBase


class AgnoTool(AnyToolBase[Tool]):
    """Wrapper class for the Tools used by Agno."""

    framework: Literal[AgentFramework.AGNO] = AgentFramework.AGNO

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> Tool:
        # Agno lets you pass callables directly in as tools ❤️
        return tool

    @property
    def name(self) -> str:
        """Returns the name of the tool."""
        return self.tool.__name__
