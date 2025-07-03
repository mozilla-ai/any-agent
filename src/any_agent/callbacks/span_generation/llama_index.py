# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools import ToolMetadata

    from any_agent.callbacks.context import Context


class _LlamaIndexSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        llm_input: list[ChatMessage] = args[1]
        input_messages = [
            {
                "role": message.role.value,
                "content": message.content or "No content",
            }
            for message in llm_input
        ]
        model_id = context.shared["model_id"]

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        agent_output: AgentOutput = args[0]

        output: str | list[dict[str, Any]] = ""
        if response := agent_output.response:
            if content := response.content:
                output = content

        if tool_calls := agent_output.tool_calls:
            output = [
                {
                    "tool.name": getattr(tool_call, "tool_name", "No name"),
                    "tool.args": getattr(tool_call, "tool_kwars", "{}"),
                }
                for tool_call in tool_calls
            ]

        input_tokens = 0
        output_tokens = 0
        raw: dict[str, Any] | None
        if raw := getattr(agent_output, "raw", None):
            token_usage: dict[str, int]
            if token_usage := raw.get("usage"):
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        meta: ToolMetadata = context.shared["metadata"]

        return self._set_tool_input(
            context, name=str(meta.name), description=meta.description, args=kwargs
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        output = args[0]

        if raw_output := getattr(output, "raw_output", None):
            if content := getattr(raw_output, "content", None):
                return self._set_tool_output(context, content[0].text)
            return self._set_tool_output(context, raw_output)
        return self._set_tool_output(context, output)
