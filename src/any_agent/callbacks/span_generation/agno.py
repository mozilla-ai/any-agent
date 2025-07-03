# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from agno.models.message import Message, MessageMetrics
    from agno.tools.function import FunctionCall

    from any_agent.callbacks.context import Context


class _AgnoSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs):
        messages: list[Message] = kwargs.get("messages", [])
        input_messages = [
            {"role": message.role, "content": str(message.content)}
            for message in messages
        ]
        return self._set_llm_input(context, context.shared["model_id"], input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
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

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        function_call: FunctionCall = args[0]
        function = function_call.function

        return self._set_tool_input(
            context,
            name=function.name,
            description=function.description,
            args=function_call.arguments,
            call_id=function_call.call_id,
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        function_call: FunctionCall = args[1]
        return self._set_tool_output(context, function_call.result)
