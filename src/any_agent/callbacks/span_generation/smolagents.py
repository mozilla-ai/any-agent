# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer
    from smolagents.models import ChatMessage
    from smolagents.tools import Tool


def _set_llm_input(messages: list[ChatMessage], span: Span) -> None:
    if not messages:
        return
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.role.value,  # type: ignore[attr-defined]
                    "content": message.content[0]["text"],  # type: ignore[index]
                }
                for message in messages
                if message.content
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: ChatMessage, span: Span) -> None:
    if content := response.content:
        span.set_attributes(
            {
                "gen_ai.output": str(content),
                "gen_ai.output.type": "text",
            }
        )
    if tool_calls := response.tool_calls:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": tool_call.function.name,
                            "tool.args": tool_call.function.arguments,
                        }
                        for tool_call in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )

    if raw := response.raw:
        if token_usage := raw.get("usage", None):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )

        if response_model := raw.get("model", None):
            span.set_attribute("gen_ai.response.model", response_model)


class _SmolagentsSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

    def before_llm_call(self, context, *args, **kwargs):
        tracer: Tracer = context["tracer"]
        model_id = context["model_id"]

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
            _set_llm_input(args[0], span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(self, context, output, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]
        _set_llm_output(output, span)
        span.set_status(StatusCode.OK)

        return context

    def before_tool_execution(self, context, *args, **kwargs):
        tracer: Tracer = context["tracer"]
        tool: Tool = context["original_tool"]

        span: Span = tracer.start_span(
            name=f"execute_tool {tool.name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.description": tool.description,
                "gen_ai.tool.args": json.dumps(kwargs),
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
