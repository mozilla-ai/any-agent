# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.google import GoogleAgent


class _GoogleADKWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original: dict[str, Any] = {}

    async def wrap(self, agent: GoogleAgent) -> None:
        self._original["before_model"] = agent._agent.before_model_callback

        def before_model_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            if callable(self._original["before_model"]):
                return self._original["before_model"](*args, **kwargs)

            return None

        agent._agent.before_model_callback = before_model_callback

        self._original["after_model"] = agent._agent.after_model_callback

        def after_model_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, *args, **kwargs)

            if callable(self._original["after_model"]):
                return self._original["after_model"](*args, **kwargs)

            return None

        agent._agent.after_model_callback = after_model_callback

        self._original["before_tool"] = agent._agent.before_tool_callback

        def before_tool_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            if callable(self._original["before_tool"]):
                return self._original["before_tool"](*args, **kwargs)

            return None

        agent._agent.before_tool_callback = before_tool_callback

        self._original["after_tool"] = agent._agent.after_tool_callback

        def after_tool_callback(*args, **kwarg) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context, *args, **kwarg)

            if callable(self._original["after_tool"]):
                return self._original["after_tool"](*args, **kwarg)

            return None

        agent._agent.after_tool_callback = after_tool_callback

    async def unwrap(self, agent: GoogleAgent) -> None:
        if "before_model" in self._original:
            agent._agent.before_model_callback = self._original["before_model"]
        if "before_tool" in self._original:
            agent._agent.before_tool_callback = self._original["before_tool"]
        if "after_model" in self._original:
            agent._agent.after_model_callback = self._original["after_model"]
        if "after_tool" in self._original:
            agent._agent.after_tool_callback = self._original["after_tool"]
