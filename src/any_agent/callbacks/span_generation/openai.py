# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from agents import FunctionTool, ModelResponse

    from any_agent.callbacks.context import Context


class _OpenAIAgentsSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        model_id = context.shared["model_id"]

        user_input = kwargs.get("input", ["No input"])[0]
        system_instructions = kwargs.get("system_instructions")
        input_messages = [
            {"role": "system", "content": system_instructions},
            user_input,
        ]
        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseOutputText,
        )

        response: ModelResponse = args[0]
        if not response.output:
            return context

        output: str | list[dict[str, Any]] = ""
        if isinstance(response.output[0], ResponseFunctionToolCall):
            output = [
                {
                    "tool.name": response.output[0].name,
                    "tool.args": response.output[0].arguments,
                }
            ]
        elif isinstance(response.output[0], ResponseOutputMessage):
            if content := response.output[0].content:
                if isinstance(content[0], ResponseOutputText):
                    output = content[0].text

        input_tokens = 0
        output_tokens = 0
        if token_usage := response.usage:
            input_tokens = token_usage.input_tokens
            output_tokens = token_usage.output_tokens

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool: FunctionTool = context.shared["original_tool"]

        return self._set_tool_input(
            context, name=tool.name, description=tool.description, args=args[1]
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        return self._set_tool_output(context, args[0])
