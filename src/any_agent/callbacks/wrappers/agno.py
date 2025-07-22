# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

from any_agent.callbacks.context import Message, ToolCall

if TYPE_CHECKING:
    from agno.models.message import Message as AgnoMessage

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.agno import AgnoAgent


def _get_context_messages(after_llm_call: bool, **kwargs) -> list[Message]:
    messages = []
    agno_messages: list[AgnoMessage] = kwargs["messages"]
    if after_llm_call:
        if assistant_message := kwargs.get("assistant_message"):
            agno_messages.append(assistant_message)
    for message in agno_messages:
        role = message.role
        content = message.content
        if not content:
            agno_tool_calls = message.tool_calls
            tool_calls = []
            for tool_call in agno_tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call["function"]["name"],
                        args=tool_call["function"]["arguments"],
                        id=tool_call["id"]
                    )
                )
            content = tool_calls
        messages.append(
            Message(
                role=role, content=content, id=id(message)
            )
        )
    return messages

def _set_framework_messages(context_messages: list[Message], **kwargs) -> list[AgnoMessage]:
    agno_messages: list[AgnoMessage] = kwargs["messages"]
    context_messages = {
        message.id: message for message in context_messages
    }
    processed_messages: list[AgnoMessage] = []

    for agno_message in agno_messages:
        # User has removed the message in callbacks
        if id(agno_message) not in context_messages:
            continue

        context_message = context_messages[id(agno_message)]
        agno_message.role = context_message.role
        if isinstance(context_message.content, str):
            agno_message.content = context_message.content
        else:
            agno_message.tool_calls = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.args
                    }
                }
                for tool_call in context_message.content
            ]

        processed_messages.append(agno_message)

    kwargs["messages"] = processed_messages
    return kwargs


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

            context.messages = _get_context_messages(after_llm_call=False, **kwargs)
            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, **kwargs)
            kwargs = _set_framework_messages(context.messages, **kwargs)

            await self._original_aprocess_model(*args, **kwargs)

            context.messages = _get_context_messages( after_llm_call=True, **kwargs)
            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, **kwargs)
            kwargs = _set_framework_messages(context.messages, **kwargs)

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
