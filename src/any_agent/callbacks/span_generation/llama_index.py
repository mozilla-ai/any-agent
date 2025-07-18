# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_agent.callbacks.span_generation.base import _SpanGeneration

if TYPE_CHECKING:
    from litellm.types.utils import ChatCompletionMessageToolCall, Usage
    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools import ToolMetadata

    from any_agent.callbacks.context import Context


class _LlamaIndexSpanGeneration(_SpanGeneration):
    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        # Handle direct dict format (from call_model wrapper)
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            input_messages = kwargs["messages"]
            if not input_messages:
                return context
            model_id = kwargs.get("model", "No model")
        # Handle LlamaIndex callback format
        elif len(args) >= 2 and isinstance(args[1], list):
            llm_input: list[ChatMessage] = args[1]
            input_messages = [
                {
                    "role": message.role.value,
                    "content": message.content or "No content",
                }
                for message in llm_input
            ]
            model_id = context.shared["model_id"]
        else:
            return context

        return self._set_llm_input(context, model_id, input_messages)

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response = args[0]
        token_usage: Usage | None
        # Handle litellm ModelResponse (from call_model wrapper)
        if hasattr(response, "choices") and hasattr(response, "model_extra"):
            if not response.choices:
                return context

            message = getattr(response.choices[0], "message", None)
            if not message:
                return context

            output: str | list[dict[str, Any]] = ""
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

            if token_usage := getattr(response, "model_extra", {}).get("usage"):
                if token_usage:
                    input_tokens = token_usage.prompt_tokens
                    output_tokens = token_usage.completion_tokens

        # Handle LlamaIndex AgentOutput
        elif hasattr(response, "response") or hasattr(response, "tool_calls"):
            agent_output: AgentOutput = response

            output = ""
            if agent_response := agent_output.response:
                if content := agent_response.content:
                    output = content

            # Fix type annotation issue - agent_output.tool_calls returns different type than ChatCompletionMessageToolCall
            if agent_tool_calls := agent_output.tool_calls:
                output = [
                    {
                        "tool.name": getattr(tool_call, "tool_name", "No name"),
                        "tool.args": getattr(tool_call, "tool_kwars", "{}"),
                    }
                    for tool_call in agent_tool_calls
                ]

            input_tokens = 0
            output_tokens = 0
            raw: dict[str, Any] | None
            if raw := getattr(agent_output, "raw", None):
                # Rename to avoid variable redefinition
                usage_info: dict[str, int]
                if usage_info := raw.get("usage"):
                    input_tokens = usage_info.get("prompt_tokens", 0)
                    output_tokens = usage_info.get("completion_tokens", 0)
        else:
            return context

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
