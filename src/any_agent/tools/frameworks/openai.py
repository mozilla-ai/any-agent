from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

with suppress(ImportError):
    from agents import Tool as AgentTool


class OpenAITool(AnyToolBase["AgentTool"]):
    """Wrapper class for the Tools used by OpenAI."""

    framework: Literal[AgentFramework.OPENAI] = AgentFramework.OPENAI

    @classmethod
    def _validate_tool_type(cls, tool: "AgentTool| Callable[..., Any]") -> "AgentTool":
        from agents import function_tool

        if isinstance(tool, AgentTool):  # type: ignore[arg-type, misc]
            return tool  # type: ignore[return-value]

        return function_tool(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self._tool.name
