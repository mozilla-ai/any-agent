# mypy: disable-error-code="no-untyped-def,override,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from any_agent.callbacks.context import Context


class _GoogleSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        llm_request: LlmRequest = kwargs["llm_request"]

        messages = []
        if config := llm_request.config:
            messages.append(
                {
                    "role": "system",
                    "content": getattr(config, "system_instruction", "No instructions"),
                }
            )
        if parts := llm_request.contents[0].parts:
            messages.append(
                {
                    "role": getattr(llm_request.contents[0], "role", "No role"),
                    "content": getattr(parts[0], "text", "No content"),
                }
            )

        return self._set_llm_input(context, str(llm_request.model), messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        llm_response: LlmResponse = kwargs["llm_response"]

        content = llm_response.content
        output: str | list[dict[str, Any]]
        if not content or not content.parts:
            output = ""
        elif content.parts[0].text:
            output = str(content.parts[0].text)
        else:
            output = [
                {
                    "tool.name": getattr(part.function_call, "name", "No name"),
                    "tool.args": getattr(part.function_call, "args", "{}"),
                }
                for part in content.parts
                if part.function_call
            ]
        input_tokens = 0
        output_tokens = 0
        if resp_meta := llm_response.usage_metadata:
            if prompt_tokens := resp_meta.prompt_token_count:
                input_tokens = prompt_tokens
            if candidates_token := resp_meta.candidates_token_count:
                output_tokens = candidates_token
        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool: BaseTool = kwargs["tool"]
        tool_args: dict[str, Any] = kwargs["args"]
        tool_context: ToolContext = kwargs["tool_context"]

        return self._set_tool_input(
            context,
            tool.name,
            tool.description,
            tool_args,
            tool_context.function_call_id,
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        return self._set_tool_output(context, kwargs["tool_response"])
