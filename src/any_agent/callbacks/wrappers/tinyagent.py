# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

import asyncio
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

            if "messages" in kwargs:
                context.framework_state.messages = kwargs["messages"]

                def get_messages():
                    return context.framework_state.messages

                def set_messages(messages):
                    context.framework_state.messages = messages
                    kwargs["messages"] = messages

                context.framework_state._message_getter = get_messages
                context.framework_state._message_setter = set_messages

            for callback in agent.config.callbacks:
                result = callback.before_llm_call(context, **kwargs)
                if asyncio.iscoroutinefunction(callback.before_llm_call):
                    context = await result
                else:
                    context = result

            output = await self._original_llm_call(**kwargs)

            for callback in agent.config.callbacks:
                result = callback.after_llm_call(context, output)
                if asyncio.iscoroutinefunction(callback.after_llm_call):
                    context = await result
                else:
                    context = result

            context.framework_state._message_getter = None
            context.framework_state._message_setter = None

            return output

        agent.call_model = wrap_call_model

        async def wrapped_tool_execution(original_call, request):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                result = callback.before_tool_execution(context, request)
                if asyncio.iscoroutinefunction(callback.before_tool_execution):
                    context = await result
                else:
                    context = result

            output = await original_call(request)

            for callback in agent.config.callbacks:
                result = callback.after_tool_execution(context, output)
                if asyncio.iscoroutinefunction(callback.after_tool_execution):
                    context = await result
                else:
                    context = result

            return output

        class WrappedCallTool:
            def __init__(self, original_call_tool):
                self.original_call_tool = original_call_tool

            async def call_tool(self, request: dict[str, Any]):
                return await wrapped_tool_execution(self.original_call_tool, request)

        self._original_clients = deepcopy(agent.clients)
        wrapped_tools = {}
        for key, tool in agent.clients.items():
            wrapped = WrappedCallTool(tool.call_tool)
            tool.call_tool = wrapped.call_tool
            wrapped_tools[key] = tool
        agent.clients = wrapped_tools

    async def unwrap(self, agent: TinyAgent) -> None:
        if self._original_llm_call:
            agent.call_model = self._original_llm_call
        if self._original_clients:
            agent.clients = self._original_clients
