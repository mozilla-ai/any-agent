# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span, StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.llms.llm import ToolSelection
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(messages: list[ChatMessage], span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.role.value,
                    "content": message.content or "No content",
                }
                for message in messages
            ]
        ),
    )


def _set_llm_output(output: AgentOutput, span: Span) -> None:
    if response := output.response:
        if content := response.content:
            span.set_attributes(
                {
                    "gen_ai.output": content,
                    "gen_ai.output.type": "text",
                }
            )
    tool_calls: list[ToolSelection]
    if tool_calls := output.tool_calls:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(tool_call, "tool_name", "No name"),
                            "tool.args": getattr(tool_call, "tool_kwars", "{}"),
                        }
                        for tool_call in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )
    raw: dict[str, Any] | None
    if raw := getattr(output, "raw", None):
        token_usage: dict[str, int]
        if token_usage := raw.get("usage"):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.get("prompt_tokens", 0),
                    "gen_ai.usage.output_tokens": token_usage.get(
                        "completion_tokens", 0
                    ),
                }
            )


class _LlamaIndexSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        super().__init__()

    def before_llm_call(
        self, context, ctx, llm_input: list[ChatMessage], tools, memory
    ):
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
            _set_llm_input(llm_input, span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(self, context, output: AgentOutput):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]
        _set_llm_output(output, span)
        span.set_status(StatusCode.OK)

        return context

    def before_tool_execution(
        self,
        context: dict[str, Any],
        *args,
        **kwargs,
    ):
        tracer: Tracer = context["tracer"]
        name = context["tool_name"]

        span = tracer.start_span(
            name=f"execute_tool {name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.args": json.dumps(kwargs, default=str, ensure_ascii=False),
            }
        )

        trace_id = span.get_span_context().trace_id
        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(self, context, output, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"execute_tool-{trace_id}"]

        if raw_output := getattr(output, "raw_output", None):
            if content := getattr(raw_output, "content", None):
                _set_tool_output(content[0].text, span)
            else:
                _set_tool_output(raw_output, span)
        else:
            _set_tool_output(output, span)

        return context
