# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from litellm.types.utils import ChatCompletionMessageToolCall, Usage

    from any_agent.callbacks.context import Context


class _LangchainSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        # Handle direct dict format (from call_model wrapper)
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            input_messages = [kwargs["messages"][-1]]
            if not input_messages:
                return context
            model_id = kwargs.get("model", "No model")
        # Handle LangChain callback format
        elif len(args) >= 2 and isinstance(args[1], list):
            messages: list[list[BaseMessage]] = args[1]
            if not messages or not messages[0]:
                return context

            input_messages = [
                {
                    "role": str(message.type).replace("human", "user"),
                    "content": str(message.content),
                }
                for message in messages[0]
            ]
            model_id = kwargs.get("invocation_params", {}).get("model", "No model")
        else:
            msg = "Unexpected Scenario"
            raise ValueError(msg)

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response = args[0]

        # Handle litellm ModelResponse (from call_model wrapper)
        if hasattr(response, "choices") and hasattr(response, "model_extra"):
            if not response.choices:
                return context

            message = getattr(response.choices[0], "message", None)
            if not message:
                return context

            output: str | list[dict[str, Any]]
            if content := getattr(message, "content", None):
                output = content

            tool_calls: list[ChatCompletionMessageToolCall] | None
            if tool_calls := getattr(message, "tool_calls", None):
                output = [
                    {
                        "tool.name": getattr(tool_call.function, "name", "No name"),
                        "tool.args": getattr(
                            tool_call.function, "arguments", "No name"
                        ),
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

        # Handle LangChain LLMResult
        elif hasattr(response, "generations"):
            if not response.generations or not response.generations[0]:
                return context

            generation = response.generations[0][0]

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
        else:
            return context

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
