# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


class Callback:
    """Base class for AnyAgent callbacks."""

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        """Will be called before any LLM Call starts."""
        return context

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        """Will be called before any Tool Execution starts."""
        return context

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        """Will be called after any LLM Call is completed."""
        return context

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        """Will be called after any Tool Execution is completed."""
        return context
