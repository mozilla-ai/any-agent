# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from agno.tools.function import FunctionCall

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
                context = await callback.before_llm_call(context, *args, **kwargs)

            result = await self._original_aprocess_model(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = await callback.after_llm_call(
                    context, result, *args, **kwargs
                )

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

            # Extract (pre) tool information
            function_call: FunctionCall = args[0]
            function = function_call.function
            current_tool_call: dict[str, Any] = {}
            current_tool_call["name"] = function.name
            current_tool_call["description"] = function.description
            current_tool_call["args"] = function_call.arguments
            current_tool_call["call_id"] = function_call.call_id
            context.shared["current_tool_call"] = current_tool_call

            for callback in agent.config.callbacks:
                context = await callback.before_tool_execution(context, *args, **kwargs)

            result = await self._original_arun_function_call(*args, **kwargs)

            # Extract (post) tool information
            current_tool_call["result"] = result[2].result

            for callback in agent.config.callbacks:
                context = await callback.after_tool_execution(context)

            return result

        agent._agent.model.arun_function_call = wrapped_tool_execution

    async def unwrap(self, agent: AgnoAgent):
        if self._original_aprocess_model is not None:
            agent._agent.model._aprocess_model_response = self._original_aprocess_model
        if self._original_arun_function_call is not None:
            agent._agent.model.arun_function_calls = self._original_arun_function_call
