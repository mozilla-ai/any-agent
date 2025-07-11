# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from any_agent.callbacks.context import Context


class _LangchainSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        # Handle both cases: messages as positional args (normal LangChain callback)
        # or messages as keyword args (when called via call_model wrapper)
        messages: list[list[BaseMessage]] | list[dict[str, Any]] | None = None

        if len(args) > 1:
            # Normal LangChain callback case
            messages = args[1]
        elif "messages" in kwargs:
            # call_model wrapper case - messages are already in dict format
            messages = kwargs["messages"]

        if not messages:
            return context

        model_id = kwargs.get("invocation_params", {}).get("model") or kwargs.get(
            "model", "No model"
        )

        # Handle both BaseMessage objects and dict messages
        if messages and isinstance(messages[0], dict):
            # Messages are already in dict format (from call_model)
            input_messages = messages
        else:
            # Messages are BaseMessage objects (from normal LangChain callback)
            input_messages = [
                {
                    "role": str(message.type).replace("human", "user"),
                    "content": str(message.content),
                }
                for message in messages[0]
            ]

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        if not args:
            return context

        response = args[0]

        # Handle LangChain LLMResult vs litellm response
        if hasattr(response, "generations") and response.generations:
            # LangChain LLMResult case
            if not response.generations[0]:
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
        if hasattr(response, "choices") and response.choices:
            # litellm response case (from call_model wrapper)
            choice = response.choices[0]
            message = choice.message

            output: str | list[dict[str, Any]]
            if hasattr(message, "tool_calls") and message.tool_calls:
                output = [
                    {
                        "tool.name": tool_call.function.name,
                        "tool.args": tool_call.function.arguments,
                    }
                    for tool_call in message.tool_calls
                ]
            else:
                output = message.content or message.get("content", "")

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            return self._set_llm_output(context, output, input_tokens, output_tokens)

        return context

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
