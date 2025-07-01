# mypy: disable-error-code="method-assign,no-untyped-def"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        ModelResponse,
        Usage,
    )
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(messages: list[dict[str, str]], span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages", json.dumps(messages, default=str, ensure_ascii=False)
    )


def _set_llm_output(response: ModelResponse, span: Span) -> None:
    if not response.choices:
        return

    message = getattr(response.choices[0], "message", None)
    if not message:
        return

    if content := getattr(message, "content", None):
        span.set_attributes(
            {
                "gen_ai.output": content,
                "gen_ai.output.type": "text",
            }
        )
    tool_calls: list[ChatCompletionMessageToolCall] | None
    if tool_calls := getattr(message, "tool_calls", None):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(tool_call.function, "name", "No name"),
                            "tool.args": getattr(
                                tool_call.function, "arguments", "No name"
                            ),
                        }
                        for tool_call in tool_calls
                        if tool_call.function
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )

    token_usage: Usage | None
    if token_usage := getattr(response, "model_extra", {}).get("usage"):
        if token_usage:
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )


class _TinyAgentSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

    def before_llm_call(self, context, *args, **kwargs):
        tracer: Tracer = context["tracer"]

        model_id = kwargs.get("model", "No model")

        span: Span = tracer.start_span(
            name=f"call_llm {model_id}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model_id,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(kwargs["messages"], span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(self, context, response, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]
        if response_model := getattr(response, "model", None):
            span.set_attribute("gen_ai.response.model", response_model)

        _set_llm_output(response, span)
        span.set_status(StatusCode.OK)

        return context

    def before_tool_execution(self, context, request: dict[str, Any], *args, **kwargs):
        tracer: Tracer = context["tracer"]

        tool_name = request.get("name", "No name")
        span: Span = tracer.start_span(
            name=f"execute_tool {tool_name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.args": json.dumps(
                    request.get("arguments", {}),
                    default=str,
                    ensure_ascii=False,
                ),
            }
        )

        trace_id = span.get_span_context().trace_id
        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(self, context, output, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"execute_tool-{trace_id}"]
        _set_tool_output(output, span)
        span.set_status(StatusCode.OK)

        return context
