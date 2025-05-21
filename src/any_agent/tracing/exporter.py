from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.logging import logger

from .agent_trace import AgentSpan, AgentTrace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan

    from any_agent import AgentFramework, TracingConfig


class AnyAgentExporter(SpanExporter):
    """Build an `AgentTrace` and export to the different outputs."""

    def __init__(  # noqa: D107
        self,
        agent_framework: AgentFramework,
        tracing_config: TracingConfig,
    ):
        self.agent_framework = agent_framework
        self.tracing_config = tracing_config
        self.traces: dict[int, AgentTrace] = {}
        self.console: Console | None = None
        self.run_trace_mapping: dict[str, int] = {}

        if self.tracing_config.console:
            self.console = Console()

    def print_to_console(self, span: AgentSpan) -> None:
        """Print the span to the console."""
        if not self.console:
            msg = "Console is not initialized"
            raise RuntimeError(msg)
        style = getattr(self.tracing_config, span.kind.lower(), None)
        if not style:
            return

        self.console.rule(span.kind, style=style)

        for key, value in span.attributes.items():
            if key in ("genai.input", "genai.output"):
                self.console.print(
                    Panel(
                        Markdown(str(value or "")),
                        title=key,
                    ),
                )
            else:
                self.console.print(f"{key}: {value}")

        self.console.rule(style=style)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: D102
        for readable_span in spans:
            # Check if this span belongs to our run
            if scope := readable_span.instrumentation_scope:
                if scope.name != "any_agent":
                    continue
            if not readable_span.attributes:
                continue
            agent_run_id = readable_span.attributes.get("gen_ai.request.id")
            trace_id = readable_span.context.trace_id
            if agent_run_id is not None:
                assert isinstance(agent_run_id, str)
                self.run_trace_mapping[agent_run_id] = trace_id
            span = AgentSpan.from_readable_span(readable_span)
            if not self.traces.get(trace_id):
                self.traces[trace_id] = AgentTrace()
            try:
                if span.attributes.get("gen_ai.operation.name") == "call_llm":
                    span.add_cost_info()

                self.traces[trace_id].add_span(span)

                if self.tracing_config.console and self.console:
                    self.print_to_console(span)

            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning("Failed to parse span data, %s, %s", span, e)
                continue
        return SpanExportResult.SUCCESS

    def pop_trace(
        self,
        agent_run_id: str,
    ) -> AgentTrace:
        """Pop the trace for the given agent run ID."""
        trace_id = self.run_trace_mapping.pop(agent_run_id, None)
        if trace_id is None:
            msg = f"Trace ID not found for agent run ID: {agent_run_id}"
            raise ValueError(msg)
        trace = self.traces.pop(trace_id, None)
        if trace is None:
            msg = f"Trace not found for trace ID: {trace_id}"
            raise ValueError(msg)
        return trace
