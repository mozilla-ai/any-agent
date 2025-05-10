from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

with suppress(ImportError):
    from llama_index.core.tools import FunctionTool as LlamaIndexToolBase


class LlamaIndexTool(AnyToolBase["LlamaIndexToolBase"]):
    """Wrapper class for the Tools used by Llama Index."""

    framework: Literal[AgentFramework.LLAMA_INDEX] = AgentFramework.LLAMA_INDEX

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        self.__name__ = self.name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the inner tool with the same parameters."""
        return self.tool(*args, **kwargs)

    @classmethod
    def _validate_tool_type(cls, tool: "LlamaIndexToolBase | Callable[..., Any]") -> "LlamaIndexToolBase":
        from llama_index.core.tools import FunctionTool as LlamaIndexToolBase

        if isinstance(tool, LlamaIndexToolBase):
            return tool

        return LlamaIndexToolBase.from_defaults(tool)

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self._tool.metadata.name or ""
