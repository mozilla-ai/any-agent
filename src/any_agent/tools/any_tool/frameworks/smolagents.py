from __future__ import annotations

from contextlib import suppress
from functools import wraps
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool.any_tool import AnyToolBase

with suppress(ImportError):
    from smolagents import Tool as SmolagentsToolBase


class SmolagentsTool(AnyToolBase["SmolagentsToolBase"], SmolagentsToolBase):
    """Wrapper class for the Tools used by Smolagents."""

    framework: Literal[AgentFramework.SMOLAGENTS] = AgentFramework.SMOLAGENTS
    description: str | None = None
    inputs: str | None = None
    output_type: str | None = None
    forward: str | None = None

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        self.__dict__["name"] = lambda _: self.tool.name
        self.description = self.tool.description
        self.inputs = self.tool.inputs
        self.output_type = self.tool.output_type
        self.forward = self.tool.forward

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> SmolagentsToolBase:
        from smolagents import tool as smolagents_tool

        if isinstance(tool, SmolagentsToolBase):
            return tool

        # Wrapping needed until upstream changes are merged
        @wraps(tool)
        def wrapped_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            return tool(*args, **kwargs)

        return smolagents_tool(wrapped_function)
