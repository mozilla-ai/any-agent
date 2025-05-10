from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

with suppress(ImportError):
    from langchain_core.tools import BaseTool as LangchainToolBase
    from langchain_core.tools import tool as langchain_tool


class LangchainTool(AnyToolBase["LangchainToolBase"]):
    """Wrapper class for the Tools used by Langchain."""

    framework: Literal[AgentFramework.LANGCHAIN] = AgentFramework.LANGCHAIN

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        from langchain_core.tools import BaseTool as LangchainToolBase

        if isinstance(self.tool, LangchainToolBase):
            self.__name__ = self.tool.name
            self.__qualname__ = self.tool.name
            self.__doc__ = self.tool.description
        else:
            self.__name__ = self.tool.__name__
            self.__qualname__ = self.tool.__name__
            self.__doc__ = self.tool.__doc__

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Call the inner tool with the same parameters."""
        return self.tool(*args, **kwargs)

    @classmethod
    def _validate_tool_type(cls, tool: "LangchainToolBase | Callable[..., Any]") -> "LangchainToolBase":
        from langchain_core.tools import BaseTool as LangchainToolBase
        from langchain_core.tools import tool as langchain_tool

        if isinstance(tool, LangchainToolBase):
            return tool

        return langchain_tool(tool)

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self._tool.name
