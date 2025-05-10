from collections.abc import Callable
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

TinyAgentToolType = Callable[..., Any]


class TinyAgentTool(AnyToolBase[TinyAgentToolType]):
    """Wrapper class for the Tools used by TinyAgent."""

    framework: Literal[AgentFramework.TINYAGENT] = AgentFramework.TINYAGENT

    @classmethod
    def _validate_tool_type(cls, tool: TinyAgentToolType) -> TinyAgentToolType:
        # Agno lets you pass callables directly in as tools ❤️
        return tool

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Call the inner tool with the same parameters."""
        return self.tool(*args, **kwargs)
