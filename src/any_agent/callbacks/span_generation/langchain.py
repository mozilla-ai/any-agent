# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(messages: list[list[BaseMessage]], span: Span) -> None:
    if not messages:
        return
    if not messages[0]:
        return

    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.type.replace("human", "user"),
                    "content": message.content,
                }
                for message in messages[0]
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: LLMResult, span: Span) -> None:
    if not response.generations:
        return
    if not response.generations[0]:
        return

    generation = response.generations[0][0]

    if text := generation.text:
        span.set_attributes(
            {
                "gen_ai.output": text,
                "gen_ai.output.type": "text",
            }
        )
    if message := getattr(generation, "message", None):
        if tool_calls := getattr(message, "tool_calls", None):
            span.set_attributes(
                {
                    "gen_ai.output": json.dumps(
                        [
                            {
                                "tool.name": tool.get("name", "No name"),
                                "tool.args": tool.get("args", "No args"),
                            }
                            for tool in tool_calls
                        ],
                        default=str,
                        ensure_ascii=False,
                    ),
                    "gen_ai.output.type": "json",
                }
            )

    if llm_output := getattr(response, "llm_output", None):
        if token_usage := llm_output.get("token_usage", None):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )


class _LangchainSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        super().__init__()

    def before_llm_call(self, context, *args, **kwargs):
        tracer: Tracer = context["tracer"]

        messages = args[1]

        model = kwargs.get("invocation_params", {}).get("model", "No model")
        span: Span = tracer.start_span(
            name=f"call_llm {model}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model,
            }
        )
        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(messages, span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(self, context, response: LLMResult, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]
        _set_llm_output(response, span)
        span.set_status(StatusCode.OK)

        return context

    def before_tool_execution(
        self,
        context: dict[str, Any],
        serialized: dict[str, Any],
        input_str,
        *args,
        **kwargs,
    ):
        tracer: Tracer = context["tracer"]

        span: Span = tracer.start_span(
            name=f"execute_tool {serialized.get('name')}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": serialized.get("name", "No name"),
                "gen_ai.tool.description": serialized.get(
                    "description", "No description"
                ),
                "gen_ai.tool.args": json.dumps(
                    kwargs.get("inputs", {}), default=str, ensure_ascii=False
                ),
            }
        )

        trace_id = span.get_span_context().trace_id
        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(self, context, output, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"execute_tool-{trace_id}"]

        if content := getattr(output, "content", None):
            _set_tool_output(content, span)
            if tool_call_id := getattr(output, "tool_call_id", None):
                span.set_attribute("gen_ai.tool.call.id", tool_call_id)

        return context
