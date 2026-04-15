# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


class Callback:
    """Base class for AnyAgent callbacks.

    Subclass `Callback` and override any subset of the available lifecycle methods
    to observe, control, or extend agent execution without modifying the underlying
    agent implementation.

    Each hook receives a `Context` object and should return that same `Context`,
    optionally after inspecting `context.current_span` or storing shared state in
    `context.shared`.

    Common uses include logging, validation, guardrails, metrics collection, cost
    tracking, and intentional cancellation.
    """

    def before_agent_invocation(self, context: Context, *args, **kwargs) -> Context:
        """Will be called before the Agent invocation starts."""
        return context

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        """Will be called before any LLM Call starts."""
        return context

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        """Will be called before any Tool Execution starts."""
        return context

    def after_agent_invocation(self, context: Context, *args, **kwargs) -> Context:
        """Will be called once the Agent invocation ends."""
        return context

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        """Will be called after any LLM Call is completed."""
        return context

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        """Will be called after any Tool Execution is completed."""
        return context
