from contextlib import suppress
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from llama_index.core.tools import FunctionTool as LlamaIndexTool


class LlamaIndexTool(AnyToolBase["LlamaIndexTool"]):
    framework: Literal[AgentFramework.LLAMA_INDEX] = AgentFramework.LLAMA_INDEX

    def model_post_init(self, _: Any) -> None:
        self.__name__ = self.tool.metadata.name

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> "LlamaIndexTool":
        from llama_index.core.tools import FunctionTool as LlamaIndexTool

        if isinstance(tool, LlamaIndexTool):
            return tool

        return LlamaIndexTool.from_defaults(tool)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.tool.metadata.name
