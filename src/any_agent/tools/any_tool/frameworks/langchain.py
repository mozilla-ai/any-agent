from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from langchain_core.tools import BaseTool as LangchainToolBase
    from langchain_core.tools import tool as langchain_tool


class LangchainTool(AnyToolBase["LangchainToolBase"]):
    """Wrapper class for the Tools used by Langchain."""

    framework: Literal[AgentFramework.LANGCHAIN] = AgentFramework.LANGCHAIN

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        self.__name__ = self.tool.name
        self.__qualname__ = self.tool.name
        self.__doc__ = self.tool.description

    def __call__(self, *args, **kwargs) -> Any:
        """Call the inner tool with the same parameters."""
        return self.tool(*args, **kwargs)

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "LangchainToolBase":
        if isinstance(tool, LangchainToolBase):
            return tool

        return langchain_tool(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.tool.name
