from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

with suppress(ImportError):
    from google.adk.tools import FunctionTool as GoogleFunctionTool


class GoogleTool(AnyToolBase["GoogleFunctionTool"]):
    """Wrapper class for the Tools used by Google."""

    framework: Literal[AgentFramework.GOOGLE] = AgentFramework.GOOGLE

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the inner tool with the same parameters."""

        from google.adk.tools import FunctionTool as GoogleFunctionTool

        if isinstance(self.tool, GoogleFunctionTool):
            return self.tool.func(*args, **kwargs)

        return self.tool(*args, **kwargs)

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "GoogleToolBase | GoogleFunctionTool":
        if isinstance(tool, GoogleToolBase):
            return tool

        return GoogleFunctionTool(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Returns the name of the tool."""
        return self.tool.name
