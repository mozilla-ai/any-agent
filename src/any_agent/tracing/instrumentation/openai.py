from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agents.tracing import TracingProcessor, set_trace_processors
from agents.tracing.span_data import FunctionSpanData, GenerationSpanData
from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(span_data: GenerationSpanData, span: Span) -> None:
    if input_messages := span_data.input:
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps(
                input_messages,
                default=str,
                ensure_ascii=False,
            ),
        )


def _set_llm_output(span_data: GenerationSpanData, span: Span) -> None:
    if not span_data.output:
        return
    output = span_data.output[0]
    if content := output.get("content"):
        span.set_attributes(
            {
                "gen_ai.output": content,
                "gen_ai.output.type": "text",
            }
        )
    if tool_calls := output.get("tool_calls"):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": tool_call.get("function", {}).get(
                                "name", "No name"
                            ),
                            "tool.args": tool_call.get("function", {}).get(
                                "arguments", "No args"
                            ),
                        }
                        for tool_call in tool_calls
                    ]
                ),
                "gen_ai.output.type": "json",
            }
        )
    if token_usage := span_data.usage:
        span.set_attributes(
            {
                "gen_ai.usage.input_tokens": token_usage["input_tokens"],
                "gen_ai.usage.output_tokens": token_usage["output_tokens"],
            }
        )


class _OpenAIAgentsInstrumentor:
    def __init__(self) -> None:
        self.first_llm_calls: set[str] = set()

    def instrument(self, tracer: Tracer) -> None:
        first_llm_calls = self.first_llm_calls

        class AnyAgentTracingProcessor(TracingProcessor):
            def __init__(self, tracer: Tracer):
                self.tracer = tracer
                self.current_spans: dict[str, Span] = {}
                super().__init__()

            def on_trace_start(self, trace):  # type: ignore[no-untyped-def]
                pass

            def on_trace_end(self, trace):  # type: ignore[no-untyped-def]
                pass

            def on_span_start(self, span):  # type: ignore[no-untyped-def]
                span_data = span.span_data
                if isinstance(span_data, GenerationSpanData):
                    model = str(span_data.model)
                    otel_span = self.tracer.start_span(
                        name=f"call_llm {model}",
                    )
                    otel_span.set_attributes(
                        {
                            "gen_ai.operation.name": "call_llm",
                            "gen_ai.request.model": model,
                        }
                    )
                    self.current_spans[span.span_id] = otel_span
                elif isinstance(span_data, FunctionSpanData):
                    otel_span = self.tracer.start_span(
                        name=f"execute_tool {span_data.name}",
                    )
                    otel_span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": span_data.name,
                        }
                    )
                    self.current_spans[span.span_id] = otel_span

            def on_span_end(self, span):  # type: ignore[no-untyped-def]
                span_data = span.span_data
                if isinstance(span_data, GenerationSpanData):
                    otel_span = self.current_spans[span.span_id]
                    trace_id = span.trace_id
                    if trace_id not in first_llm_calls:
                        first_llm_calls.add(trace_id)
                        _set_llm_input(span_data, otel_span)
                    _set_llm_output(span_data, otel_span)
                    otel_span.set_status(StatusCode.OK)
                    otel_span.end()
                    del self.current_spans[span.span_id]
                elif isinstance(span_data, FunctionSpanData):
                    otel_span = self.current_spans[span.span_id]
                    otel_span.set_attributes(
                        {
                            "gen_ai.tool.args": span_data.input or "{}",
                        }
                    )
                    _set_tool_output(span_data.output, otel_span)
                    otel_span.set_status(StatusCode.OK)
                    otel_span.end()
                    del self.current_spans[span.span_id]

            def force_flush(self):  # type: ignore[no-untyped-def]
                pass

            def shutdown(self):  # type: ignore[no-untyped-def]
                pass

        set_trace_processors([AnyAgentTracingProcessor(tracer)])

    def uninstrument(self) -> None:
        from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

        GLOBAL_TRACE_PROVIDER.set_processors([])
