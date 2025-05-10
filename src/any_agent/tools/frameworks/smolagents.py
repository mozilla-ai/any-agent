from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import wraps
from typing import Any, Literal

from any_agent.config import AgentFramework
from any_agent.tools.any_tool import AnyToolBase

with suppress(ImportError):
    from smolagents import Tool as SmolagentsToolBase


class SmolagentsTool(AnyToolBase["SmolagentsToolBase"], SmolagentsToolBase):  # type: ignore[misc]
    """Wrapper class for the Tools used by Smolagents."""

    framework: Literal[AgentFramework.SMOLAGENTS] = AgentFramework.SMOLAGENTS
    description: str | None = None  # type: ignore[assignment]
    inputs: Mapping[str, Mapping[str, str | type | bool]] | None = None  # type: ignore[assignment]
    output_type: str | None = None  # type: ignore[assignment]
    forward: Callable[..., Any] | None = None

    def model_post_init(self, _: Any) -> None:
        """Post-init tool parameters."""
        self.__dict__["name"] = lambda _: self._tool.name
        self.description = self._tool.description
        self.inputs = self._tool.inputs
        self.output_type = self._tool.output_type
        self.forward = self._tool.forward

    @classmethod
    def _validate_tool_type(cls, tool: "SmolagentsToolBase | Callable[..., Any]") -> SmolagentsToolBase:
        from smolagents import tool as smolagents_tool

        if isinstance(tool, SmolagentsToolBase):
            return tool

        # Wrapping needed until upstream changes are merged
        @wraps(tool)
        def wrapped_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            return tool(*args, **kwargs)

        return smolagents_tool(wrapped_function)
