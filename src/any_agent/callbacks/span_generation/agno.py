# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from agno.models.message import Message, MessageMetrics
    from agno.tools.function import FunctionCall

    from any_agent.callbacks.context import Context


class _AgnoSpanGeneration(_SpanGeneration):
    async def before_llm_call(self, context: Context, *args, **kwargs):
        messages: list[Message] = kwargs.get("messages", [])
        input_messages = [
            {"role": message.role, "content": str(message.content)}
            for message in messages
        ]
        return self._set_llm_input(context, context.shared["model_id"], input_messages)

    async def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        output: str | list[dict[str, Any]] = ""
        if assistant_message := kwargs.get("assistant_message"):
            if content := getattr(assistant_message, "content", None):
                output = str(content)
            if tool_calls := getattr(assistant_message, "tool_calls", None):
                output = [
                    {
                        "tool.name": tool.get("function", {}).get("name", "No name"),
                        "tool.args": tool.get("function", {}).get(
                            "arguments", "No args"
                        ),
                    }
                    for tool in tool_calls
                ]

            metrics: MessageMetrics | None
            input_tokens: int = 0
            output_tokens: int = 0
            if metrics := getattr(assistant_message, "metrics", None):
                input_tokens = metrics.input_tokens
                output_tokens = metrics.output_tokens

            context = self._set_llm_output(context, output, input_tokens, output_tokens)

        return context

    async def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        current_tool_call = context.shared["current_tool_call"]

        return self._set_tool_input(
            context,
            name=current_tool_call["name"],
            description=current_tool_call["description"],
            args=current_tool_call["args"],
            call_id=current_tool_call["call_id"],
        )

    async def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        current_tool_call = context.shared["current_tool_call"]
        return self._set_tool_output(context, current_tool_call["result"])
