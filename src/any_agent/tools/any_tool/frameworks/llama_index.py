from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from llama_index.core.tools import FunctionTool as LlamaIndexToolBase


class LlamaIndexTool(AnyToolBase["LlamaIndexToolBase"]):
    """Wrapper class for the Tools used by Llama Index."""

    framework: Literal[AgentFramework.LLAMA_INDEX] = AgentFramework.LLAMA_INDEX

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        self.__name__ = self.tool.metadata.name

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "LlamaIndexToolBase":
        if isinstance(tool, LlamaIndexToolBase):
            return tool

        return LlamaIndexToolBase.from_defaults(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.tool.metadata.name
