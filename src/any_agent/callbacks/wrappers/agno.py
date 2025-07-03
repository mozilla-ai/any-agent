# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.agno import AgnoAgent


class _AgnoWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_aprocess_model: Any = None
        self._original_arun_function_call: Any = None

    async def wrap(self, agent: AgnoAgent) -> None:
        self._original_aprocess_model = agent._agent.model._aprocess_model_response

        async def wrapped_llm_call(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = agent._agent.model.id

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            result = await self._original_aprocess_model(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, result, *args, **kwargs)

            return result

        agent._agent.model._aprocess_model_response = wrapped_llm_call

        self._original_arun_function_call = agent._agent.model.arun_function_call

        async def wrapped_tool_execution(
            *args,
            **kwargs,
        ):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            result = await self._original_arun_function_call(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(
                    context, result, *args, **kwargs
                )

            return result

        agent._agent.model.arun_function_call = wrapped_tool_execution

    async def unwrap(self, agent: AgnoAgent):
        if self._original_aprocess_model is not None:
            agent._agent.model._aprocess_model_response = self._original_aprocess_model
        if self._original_arun_function_call is not None:
            agent._agent.model.arun_function_calls = self._original_arun_function_call
