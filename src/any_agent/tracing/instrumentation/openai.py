# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from agents import GenerationSpanData
    from agents.tracing import TracingProcessor
    from opentelemetry.trace import Span

    from any_agent.frameworks.openai import OpenAIAgent


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
        self.first_llm_calls: set[int] = set()
        self.current_spans: dict[str, Span] = {}
        self._processor: TracingProcessor | None = None

    def instrument(self, agent: OpenAIAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        from agents import FunctionSpanData, GenerationSpanData
        from agents.tracing import TracingProcessor
        from agents.tracing.processors import BatchTraceProcessor

        first_llm_calls = self.first_llm_calls
        current_spans = self.current_spans

        tracer = agent._tracer

        class AnyAgentTracingProcessor(TracingProcessor):
            def on_trace_start(self, trace):
                pass

            def on_trace_end(self, trace):
                pass

            def on_span_start(self, span):
                span_data = span.span_data
                if isinstance(span_data, GenerationSpanData):
                    model = str(span_data.model)
                    otel_span = tracer.start_span(
                        name=f"call_llm {model}",
                    )
                    otel_span.set_attributes(
                        {
                            "gen_ai.operation.name": "call_llm",
                            "gen_ai.request.model": model,
                        }
                    )
                    current_spans[span.span_id] = otel_span
                elif isinstance(span_data, FunctionSpanData):
                    otel_span = tracer.start_span(
                        name=f"execute_tool {span_data.name}",
                    )
                    otel_span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": span_data.name,
                        }
                    )
                    current_spans[span.span_id] = otel_span

            def on_span_end(self, span):
                if span.span_id not in current_spans:
                    return
                span_data = span.span_data
                if isinstance(span_data, GenerationSpanData):
                    otel_span = current_spans[span.span_id]
                    trace_id = otel_span.get_span_context().trace_id
                    if trace_id not in first_llm_calls:
                        first_llm_calls.add(trace_id)
                        _set_llm_input(span_data, otel_span)
                    _set_llm_output(span_data, otel_span)
                    otel_span.set_status(StatusCode.OK)
                    otel_span.end()
                    if trace_id in agent._running_traces:
                        agent._running_traces[trace_id].add_span(otel_span)
                    del current_spans[span.span_id]
                elif isinstance(span_data, FunctionSpanData):
                    otel_span = current_spans[span.span_id]
                    otel_span.set_attributes(
                        {
                            "gen_ai.tool.args": span_data.input or "{}",
                        }
                    )
                    _set_tool_output(span_data.output, otel_span)
                    otel_span.end()
                    trace_id = otel_span.get_span_context().trace_id
                    if trace_id in agent._running_traces:
                        agent._running_traces[trace_id].add_span(otel_span)
                    del current_spans[span.span_id]

            def force_flush(self):
                pass

            def shutdown(self):
                pass

        self._processor = AnyAgentTracingProcessor()
        from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

        with GLOBAL_TRACE_PROVIDER._multi_processor._lock:
            GLOBAL_TRACE_PROVIDER._multi_processor._processors = tuple(
                p
                for p in GLOBAL_TRACE_PROVIDER._multi_processor._processors
                if not isinstance(p, BatchTraceProcessor)
            )
            GLOBAL_TRACE_PROVIDER._multi_processor._processors += (self._processor,)

    def uninstrument(self, agent: OpenAIAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

        if self._processor:
            with GLOBAL_TRACE_PROVIDER._multi_processor._lock:
                GLOBAL_TRACE_PROVIDER._multi_processor._processors = tuple(
                    p
                    for p in GLOBAL_TRACE_PROVIDER._multi_processor._processors
                    if p is not self._processor
                )
