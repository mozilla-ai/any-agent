# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from any_agent.frameworks.agno import AgnoAgent


class _AgnoWrapper:
    def __init__(self) -> None:
        self._original_aprocess_model: Any = None
        self._original_arun_function_call: Any = None
        self.context: dict[str, Any] = {}

    async def wrap(self, agent: AgnoAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        self.context["running_traces"] = agent._running_traces
        self.context["tracer"] = agent._tracer
        self.context["model_id"] = agent._agent.model.id

        self._original_aprocess_model = agent._agent.model._aprocess_model_response

        async def wrapped_llm_call(*args, **kwargs):
            for callback in agent.config.callbacks:
                self.context = callback.before_llm_call(self.context, *args, **kwargs)

            result = await self._original_aprocess_model(*args, **kwargs)

            for callback in agent.config.callbacks:
                self.context = callback.after_llm_call(
                    self.context, result, *args, **kwargs
                )

            return result

        agent._agent.model._aprocess_model_response = wrapped_llm_call

        self._original_arun_function_call = agent._agent.model.arun_function_call

        async def wrapped_tool_execution(
            *args,
            **kwargs,
        ):
            for callback in agent.config.callbacks:
                self.context = callback.before_tool_execution(
                    self.context, *args, **kwargs
                )

            result = await self._original_arun_function_call(*args, **kwargs)

            for callback in agent.config.callbacks:
                self.context = callback.after_tool_execution(
                    self.context, result, *args, **kwargs
                )

            return result

        agent._agent.model.arun_function_call = wrapped_tool_execution

    async def unwrap(self, agent: AgnoAgent):
        if len(agent._running_traces) > 1:
            return
        model = agent._agent.model
        if self._original_aprocess_model is not None:
            model._aprocess_model_response = self._original_aprocess_model
        if self._original_arun_function_call is not None:
            model.arun_function_calls = self._original_arun_function_call
