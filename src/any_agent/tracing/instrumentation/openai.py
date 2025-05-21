from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class _OpenAIAgentsInstrumentor:
    def instrument(self, tracer: Tracer) -> None:
        from agents.tracing import TracingProcessor, set_trace_processors
        from agents.tracing.span_data import FunctionSpanData, GenerationSpanData

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
                    if output := span_data.output:
                        if content := output[0].get("content"):
                            otel_span.set_attributes(
                                {
                                    "genai.output": content,
                                    "genai.output.type": "text",
                                }
                            )
                        elif tool_calls := output[0].get("tool_calls"):
                            # Tool Call
                            otel_span.set_attributes(
                                {
                                    "genai.output": json.dumps(
                                        tool_calls,
                                        default=str,
                                        ensure_ascii=False,
                                    ),
                                    "genai.output.type": "json",
                                }
                            )
                    if token_usage := span_data.usage:
                        otel_span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": token_usage[
                                    "input_tokens"
                                ],
                                "gen_ai.usage.output_tokens": token_usage[
                                    "output_tokens"
                                ],
                            }
                        )
                    otel_span.set_status(StatusCode.OK)
                    otel_span.end()
                    del self.current_spans[span.span_id]
                elif isinstance(span_data, FunctionSpanData):
                    otel_span = self.current_spans[span.span_id]
                    otel_span.set_attributes(
                        {
                            "genai.output": span_data.output or "no_output",
                            "genai.output.type": "json",
                        }
                    )
                    otel_span.set_status(StatusCode.OK)
                    otel_span.end()
                    del self.current_spans[span.span_id]

            def force_flush(self):  # type: ignore[no-untyped-def]
                pass

            def shutdown(self):  # type: ignore[no-untyped-def]
                pass

        set_trace_processors([AnyAgentTracingProcessor(tracer)])

    def uninstrument(self) -> None:
        pass
