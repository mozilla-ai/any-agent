# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from agents import FunctionTool, ModelResponse
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(system_instructions, user_input, span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [{"role": "system", "content": system_instructions}, user_input],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: ModelResponse, span: Span) -> None:
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    if not response.output:
        return
    output = response.output[0]
    if isinstance(output, ResponseFunctionToolCall):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": output.name,
                            "tool.args": output.arguments,
                        }
                    ]
                ),
                "gen_ai.output.type": "json",
            }
        )
    elif isinstance(output, ResponseOutputMessage):
        if content := output.content:
            if isinstance(content[0], ResponseOutputText):
                span.set_attributes(
                    {
                        "gen_ai.output": content[0].text,
                        "gen_ai.output.type": "text",
                    }
                )

    if token_usage := response.usage:
        span.set_attributes(
            {
                "gen_ai.usage.input_tokens": token_usage.input_tokens,
                "gen_ai.usage.output_tokens": token_usage.output_tokens,
            }
        )


class _OpenAIAgentsSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

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
            system_instructions = kwargs.get("system_instructions")
            if user_input := kwargs.get("input"):
                _set_llm_input(system_instructions, user_input[0], span)

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
        tool: FunctionTool = context["original_tool"]
        span = tracer.start_span(
            name=f"execute_tool {tool.name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.description": tool.description,
                "gen_ai.tool.args": args[1],
            }
        )
        trace_id = span.get_span_context().trace_id

        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(self, context, output, *args, **kwargs):
        trace_id = get_current_span().get_span_context().trace_id
        span = context[f"execute_tool-{trace_id}"]
        _set_tool_output(output, span)

        return context
