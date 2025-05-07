from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from langchain_core.tools import BaseTool as LangchainTool


class LangchainTool(AnyToolBase["LangchainTool"]):
    framework: Literal[AgentFramework.LANGCHAIN] = AgentFramework.LANGCHAIN

    def model_post_init(self, _: Any) -> None:
        self.__name__ = self.tool.name
        self.__qualname__ = self.tool.name
        self.__doc__ = self.tool.description

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "LangchainTool":
        from langchain_core.tools import BaseTool as LangchainTool
        from langchain_core.tools import tool as langchain_tool

        if isinstance(tool, LangchainTool):
            return tool

        return langchain_tool(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.tool.name
