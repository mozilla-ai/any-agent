from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.trace import Span, Tracer

    from any_agent.tracing.agent_trace import AgentTrace


@dataclass
class FrameworkState:
    """Framework-specific state that can be accessed and modified by callbacks.

    This object provides a consistent interface for accessing framework state across
    different agent frameworks, while the actual content is framework-specific.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Internal storage for messages. Use get_messages() and set_messages() instead."""

    _message_getter: Callable[[], list[dict[str, Any]]] | None = field(
        default=None, repr=False
    )
    """Framework-specific message getter function."""

    _message_setter: Callable[[list[dict[str, Any]]], None] | None = field(
        default=None, repr=False
    )
    """Framework-specific message setter function."""

    def get_messages(self) -> list[dict[str, Any]]:
        """Get messages in a normalized dict format.

        Returns a list of message dicts with 'role' and 'content' keys.
        Works consistently across all frameworks.

        Returns:
            List of message dicts with 'role' and 'content' keys.

        Raises:
            NotImplementedError: If the framework doesn't support message access yet.

        Example:
            ```python
            messages = context.framework_state.get_messages()
            # [{"role": "user", "content": "Hello"}]
            ```

        """
        if self._message_getter is None:
            msg = "get_messages() is not implemented for this framework yet"
            raise NotImplementedError(msg)
        return self._message_getter()

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Set messages from a normalized dict format.

        Accepts a list of message dicts with 'role' and 'content' keys and
        converts them to the framework-specific format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Raises:
            NotImplementedError: If the framework doesn't support message modification yet.

        Example:
            ```python
            messages = context.framework_state.get_messages()
            messages[-1]["content"] = "Say hello"
            context.framework_state.set_messages(messages)
            ```

        """
        if self._message_setter is None:
            msg = "set_messages() is not implemented for this framework yet"
            raise NotImplementedError(msg)
        self._message_setter(messages)


@dataclass
class Context:
    """Object that will be shared across callbacks.

    Each AnyAgent.run has a separate `Context` available.

    `shared` can be used to store and pass information
    across different callbacks.
    """

    current_span: Span
    """You can use the span in your callbacks to get information consistently across frameworks.

    You can find information about the attributes (available under `current_span.attributes`) in
    [Attributes Reference](./tracing.md#any_agent.tracing.attributes).
    """

    trace: AgentTrace
    tracer: Tracer

    shared: dict[str, Any]
    """Can be used to store arbitrary information for sharing across callbacks."""

    framework_state: FrameworkState
    """Framework-specific state that can be accessed and modified by callbacks.

    Provides consistent access to framework state across different agent frameworks.
    See [`FrameworkState`][any_agent.callbacks.context.FrameworkState] for available attributes.

    Example:
        ```python
        class ModifyPromptCallback(Callback):
            def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
                # Modify the last message content
                if context.framework_state.messages:
                    context.framework_state.messages[-1]["content"] = "Say hello"
                return context
        ```
    """
