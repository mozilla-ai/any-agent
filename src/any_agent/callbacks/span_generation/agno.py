# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from agno.models.message import Message, MessageMetrics
    from agno.tools.function import FunctionCall
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(messages: list[Message], span: Span) -> None:
    if not messages:
        return
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(assistant_message: Message, span: Span) -> None:
    if content := getattr(assistant_message, "content", None):
        span.set_attributes(
            {
                "gen_ai.output": str(content),
                "gen_ai.output.type": "text",
            }
        )
    if tool_calls := getattr(assistant_message, "tool_calls", None):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": tool.get("function", {}).get(
                                "name", "No name"
                            ),
                            "tool.args": tool.get("function", {}).get(
                                "arguments", "No args"
                            ),
                        }
                        for tool in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )
    metrics: MessageMetrics | None
    if metrics := getattr(assistant_message, "metrics", None):
        span.set_attributes(
            {
                "gen_ai.usage.input_tokens": metrics.input_tokens,
                "gen_ai.usage.output_tokens": metrics.output_tokens,
            }
        )


class _AgnoSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        super().__init__()

    def before_llm_call(self, context, *args, **kwargs):
        tracer: Tracer = context["tracer"]
        model_id = context["model_id"]

        span = tracer.start_span(f"call_llm {model_id}")
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model_id,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(kwargs.get("messages", []), span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(self, context, result, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]

        if assistant_message := kwargs.get("assistant_message"):
            _set_llm_output(assistant_message, span)
        span.set_status(StatusCode.OK)

        return context

    def before_tool_execution(self, context, *args, **kwargs):
        function_call: FunctionCall = args[0]
        function = function_call.function

        tracer: Tracer = context["tracer"]

        span: Span = tracer.start_span(
            name=f"execute_tool {function.name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": function.name,
                "gen_ai.tool.description": str(function.description),
                "gen_ai.tool.args": json.dumps(
                    function_call.arguments,
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.tool.call.id": str(function_call.call_id),
            }
        )

        trace_id = span.get_span_context().trace_id
        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(self, context, result, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span = context[f"execute_tool-{trace_id}"]

        function_call: FunctionCall = result[2]
        _set_tool_output(function_call.result, span)

        return context
