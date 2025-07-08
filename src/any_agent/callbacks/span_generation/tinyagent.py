# mypy: disable-error-code="method-assign,no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        ModelResponse,
        Usage,
    )

    from any_agent.callbacks.context import Context


class _TinyAgentSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        return self._set_llm_input(
            context,
            model_id=kwargs.get("model", "No model"),
            input_messages=kwargs.get("messages", []),
        )

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response: ModelResponse = args[0]

        if not response.choices:
            return context

        message = getattr(response.choices[0], "message", None)
        if not message:
            return context

        if content := getattr(message, "content", None):
            output = content

        tool_calls: list[ChatCompletionMessageToolCall] | None
        if tool_calls := getattr(message, "tool_calls", None):
            output = [
                {
                    "tool.name": getattr(tool_call.function, "name", "No name"),
                    "tool.args": getattr(tool_call.function, "arguments", "No name"),
                }
                for tool_call in tool_calls
                if tool_call.function
            ]

        input_tokens = 0
        output_tokens = 0
        token_usage: Usage | None
        if token_usage := getattr(response, "model_extra", {}).get("usage"):
            if token_usage:
                input_tokens = token_usage.prompt_tokens
                output_tokens = token_usage.completion_tokens

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context, *args, **kwargs):
        request: dict[str, Any] = args[0]
        return self._set_tool_input(
            context,
            name=request.get("name", "No name"),
            args=request.get("arguments", {}),
        )

    def after_tool_execution(self, context, *args, **kwargs):
        return self._set_tool_output(context, args[0])
