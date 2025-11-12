# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.agno import AgnoAgent

try:
    from agno.models.message import Message

    agno_available = True
except ImportError:
    agno_available = False
    Message = None  # type: ignore[assignment,misc]


class _AgnoWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_ainvoke: Any = None
        self._original_arun_function_call: Any = None

    async def wrap(self, agent: AgnoAgent) -> None:
        if not agno_available:
            msg = "Agno is not installed"
            raise ImportError(msg)

        self._original_ainvoke = agent._agent.model.ainvoke

        async def wrapped_ainvoke(messages, *args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = agent._agent.model.id

            def get_messages():
                return [
                    {
                        "role": msg.role,
                        "content": msg.content if msg.content else "",
                    }
                    for msg in messages
                ]

            def set_messages(new_messages):
                messages.clear()
                for msg_dict in new_messages:
                    msg = Message(
                        role=msg_dict.get("role", "user"),
                        content=msg_dict.get("content", ""),
                    )
                    if "tool_calls" in msg_dict:
                        msg.tool_calls = msg_dict["tool_calls"]
                    if "tool_call_id" in msg_dict:
                        msg.tool_call_id = msg_dict["tool_call_id"]
                    if "name" in msg_dict:
                        msg.name = msg_dict["name"]
                    messages.append(msg)

            context.framework_state._message_getter = get_messages
            context.framework_state._message_setter = set_messages

            callback_kwargs = {**kwargs, "messages": messages}
            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **callback_kwargs)

            result = await self._original_ainvoke(messages, *args, **kwargs)

            callback_kwargs = {**kwargs, "assistant_message": result}
            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, *args, **callback_kwargs)

            context.framework_state._message_getter = None
            context.framework_state._message_setter = None

            return result

        agent._agent.model.ainvoke = wrapped_ainvoke

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
        if self._original_ainvoke is not None:
            agent._agent.model.ainvoke = self._original_ainvoke
        if self._original_arun_function_call is not None:
            agent._agent.model.arun_function_call = self._original_arun_function_call
