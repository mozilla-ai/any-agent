# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.tinyagent import TinyAgent


class _TinyAgentWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_llm_call: Callable[..., Any] | None = None
        self._original_clients: Any | None = None

    async def wrap(self, agent: TinyAgent) -> None:
        self._original_llm_call = agent.call_model

        async def wrap_call_model(**kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, **kwargs)

            output = await self._original_llm_call(**kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, output)

            return output

        agent.call_model = wrap_call_model

        async def wrapped_tool_execution(original_call, request):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, request)

            output = await original_call(request)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context, output)

            return output

        class WrappedCallTool:
            def __init__(self, original_call_tool):
                self.original_call_tool = original_call_tool

            async def call_tool(self, request: dict[str, Any]):
                return await wrapped_tool_execution(self.original_call_tool, request)

        self._original_clients = deepcopy(agent.clients)
        wrapped_tools = {}
        for key, tool in agent.clients.items():
            wrapped = WrappedCallTool(tool.call_tool)  # type: ignore[no-untyped-call]
            tool.call_tool = wrapped.call_tool
            wrapped_tools[key] = tool
        agent.clients = wrapped_tools

    async def unwrap(self, agent: TinyAgent) -> None:
        if self._original_llm_call:
            agent.call_model = self._original_llm_call
        if self._original_clients:
            agent.clients = self._original_clients
