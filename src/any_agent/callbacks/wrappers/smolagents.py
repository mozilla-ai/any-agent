# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from any_agent.utils import run_async_in_sync

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.smolagents import SmolagentsAgent


class _SmolagentsWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_llm_call: Callable[..., Any] | None = None
        self._original_tools: Any | None = None

    async def wrap(self, agent: SmolagentsAgent) -> None:
        self._original_llm_call = agent._agent.model.generate

        async def wrap_generate(trace_id, *args, **kwargs):
            context = self.callback_context[
                trace_id
            ]
            context.shared["model_id"] = str(agent._agent.model.model_id)

            for callback in agent.config.callbacks:
                context = await callback.before_llm_call(context, *args, **kwargs)

            output = self._original_llm_call(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = await callback.after_llm_call(context, output)

            return output
        
        def wrap_generate_sync(*args, **kwargs):
            trace_id = get_current_span().get_span_context().trace_id
            return(run_async_in_sync(wrap_generate(trace_id, *args, **kwargs)))

        agent._agent.model.generate = wrap_generate_sync

        async def wrapped_tool_execution(trace_id, original_tool, original_call, *args, **kwargs):
            context = self.callback_context[
                trace_id
            ]
            context.shared["original_tool"] = original_tool

            for callback in agent.config.callbacks:
                context = await callback.before_tool_execution(context, *args, **kwargs)

            output = original_call(**kwargs)

            for callback in agent.config.callbacks:
                context = await callback.after_tool_execution(
                    context, output, *args, **kwargs
                )

            return output

        def wrapped_tool_execution_sync(original_tool, original_call, *args, **kwargs):
            trace_id = get_current_span().get_span_context().trace_id
            return(run_async_in_sync(wrapped_tool_execution(trace_id, original_tool, original_call, *args, **kwargs)))

        class WrappedToolCall:
            def __init__(self, original_tool, original_forward):
                self.original_tool = original_tool
                self.original_forward = original_forward

            def forward(self, *args, **kwargs):
                return wrapped_tool_execution_sync(
                    self.original_tool, self.original_forward, *args, **kwargs
                )

        self._original_tools = deepcopy(agent._agent.tools)
        wrapped_tools = {}
        for key, tool in agent._agent.tools.items():
            original_forward = tool.forward
            wrapped = WrappedToolCall(tool, original_forward)
            tool.forward = wrapped.forward
            wrapped_tools[key] = tool
        agent._agent.tools = wrapped_tools

    async def unwrap(self, agent: SmolagentsAgent) -> None:
        if self._original_llm_call is not None:
            agent._agent.model.generate = self._original_llm_call
        if self._original_tools is not None:
            agent._agent.tools = self._original_tools
