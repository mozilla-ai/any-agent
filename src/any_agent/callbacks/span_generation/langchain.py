# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from any_agent.callbacks.context import Context


class _LangchainSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        messages: list[list[BaseMessage]] = args[1]
        if not messages or not messages[0]:
            return context

        model_id = kwargs.get("invocation_params", {}).get("model", "No model")

        input_messages = [
            {
                "role": str(message.type).replace("human", "user"),
                "content": str(message.content),
            }
            for message in messages[0]
        ]

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response: LLMResult = args[0]
        if not response.generations or not response.generations[0]:
            return context

        generation = response.generations[0][0]

        output: str | list[dict[str, Any]]
        if text := generation.text:
            output = text
        elif message := getattr(generation, "message", None):
            if tool_calls := getattr(message, "tool_calls", None):
                output = [
                    {
                        "tool.name": tool.get("name", "No name"),
                        "tool.args": tool.get("args", "No args"),
                    }
                    for tool in tool_calls
                ]

        input_tokens = 0
        output_tokens = 0
        if llm_output := getattr(response, "llm_output", None):
            if token_usage := llm_output.get("token_usage", None):
                input_tokens = token_usage.prompt_tokens
                output_tokens = token_usage.completion_tokens

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        serialized: dict[str, Any] = args[0]
        return self._set_tool_input(
            context,
            name=serialized.get("name", "No name"),
            description=serialized.get("description"),
            args=kwargs.get("inputs"),
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        output = args[0]
        if content := getattr(output, "content", None):
            return self._set_tool_output(context, content)

        return context
