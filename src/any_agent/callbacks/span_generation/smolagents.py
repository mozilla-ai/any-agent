# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from smolagents.models import ChatMessage
    from smolagents.tools import Tool

    from any_agent.callbacks.context import Context


class _SmolagentsSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        model_id = context.shared["model_id"]

        messages: list[ChatMessage] = args[0]
        input_messages = [
            {
                "role": message.role.value,
                "content": message.content[0]["text"],  # type: ignore[index]
            }
            for message in messages
            if message.content
        ]

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response: ChatMessage = args[0]
        output: str | list[dict[str, Any]]
        if content := response.content:
            output = str(content)
        elif tool_calls := response.tool_calls:
            output = [
                {
                    "tool.name": tool_call.function.name,
                    "tool.args": tool_call.function.arguments,
                }
                for tool_call in tool_calls
            ]

        input_tokens = 0
        output_tokens = 0
        if raw := response.raw:
            if token_usage := raw.get("usage", None):
                input_tokens = token_usage.prompt_tokens
                output_tokens = token_usage.completion_tokens

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool: Tool = context.shared["original_tool"]

        return self._set_tool_input(
            context, name=tool.name, description=tool.description, args=kwargs
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        return self._set_tool_output(context, args[0])
