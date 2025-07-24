# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from agents import FunctionTool, ModelResponse

    from any_agent.callbacks.context import Context


class _OpenAIAgentsSpanGeneration(_SpanGeneration):
    async def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        model_id = context.shared["model_id"]

        user_input = kwargs.get("input", ["No input"])[0]
        system_instructions = kwargs.get("system_instructions")
        input_messages = [
            {"role": "system", "content": system_instructions},
            user_input,
        ]
        return self._set_llm_input(context, model_id, input_messages)

    async def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
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

    async def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool: FunctionTool = context.shared["original_tool"]

        context.shared["current_tool_call"] = {}
        context.shared["current_tool_call"]["name"] = tool.name
        context.shared["current_tool_call"]["description"] = tool.description
        context.shared["current_tool_call"]["args"] = args[1]

        print("--> Storing in tool info...")
        print(f"--> {context.shared}")

        return self._set_tool_input(
            context, name=tool.name, description=tool.description, args=args[1]
        )

    async def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool_output = args[0]
        output_type = self._determine_output_type(tool_output)
        output_attr = self._serialize_for_attribute(tool_output)
        context.shared["current_tool_call"]["output_type"] = output_type
        context.shared["current_tool_call"]["output_attr"] = output_attr
        context.shared["current_tool_call"]["status"] = self._determine_tool_status(output_attr, output_type)

        print("--> Storing out tool info...")
        print(f"--> {context.shared}")

        return self._set_tool_output(context, args[0])
