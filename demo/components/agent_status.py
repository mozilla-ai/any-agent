from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from any_agent import AgentFramework, AnyAgent
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentSpan

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


class StreamlitExporter(SpanExporter):
    """Build an `AgentTrace` and export to the different outputs."""

    def __init__(  # noqa: D107
        self, agent_framework: AgentFramework, callback: Callable
    ):
        self.agent_framework = agent_framework
        self.processor: TracingProcessor | None = TracingProcessor.create(
            agent_framework
        )
        self.callback = callback

    def export(self, spans: Sequence["ReadableSpan"]) -> SpanExportResult:  # noqa: D102
        if not self.processor:
            return SpanExportResult.SUCCESS

        for readable_span in spans:
            # Check if this span belongs to our run
            span = AgentSpan.from_readable_span(readable_span)
            self.callback(span)

        return SpanExportResult.SUCCESS


def export_logs(agent: AnyAgent, callback: Callable) -> None:
    exporter = StreamlitExporter(agent.framework, callback)
    span_processor = SimpleSpanProcessor(exporter)
    agent._tracer_provider.add_span_processor(span_processor)
