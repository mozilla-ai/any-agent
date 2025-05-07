from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from google.adk.tools import BaseTool as GoogleTool
    from google.adk.tools import FunctionTool as GoogleFunctionTool


class GoogleTool(AnyToolBase["GoogleTool | GoogleFunctionTool"]):
    framework: Literal[AgentFramework.GOOGLE] = AgentFramework.GOOGLE

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "GoogleTool | GoogleFunctionTool":
        if isinstance(tool, GoogleTool):
            return tool

        return GoogleFunctionTool(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Returns the name of the tool."""
        return self.tool.name
