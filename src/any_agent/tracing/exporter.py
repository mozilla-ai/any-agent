from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.logging import logger

from .agent_trace import AgentSpan, AgentTrace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan

    from any_agent import TracingConfig


def _get_output_panel(span: AgentSpan) -> Panel | None:
    if output := span.attributes.get("gen_ai.output", None):
        output_type = span.attributes.get("gen_ai.output.type", "text")
        return Panel(
            Markdown(output) if output_type != "json" else JSON(output),
            title="OUTPUT",
            style="white",
            title_align="left",
        )
    return None


class _AnyAgentExporter(SpanExporter):
    def __init__(
        self,
        tracing_config: TracingConfig,
    ):
        self.tracing_config = tracing_config
        self.traces: dict[int, AgentTrace] = {}
        self.console: Console | None = None
        self.run_trace_mapping: dict[str, int] = {}

        if self.tracing_config.console:
            self.console = Console()

    def print_to_console(self, span: AgentSpan) -> None:
        if not self.console:
            msg = "Console is not initialized"
            raise RuntimeError(msg)

        operation_name = span.attributes.get("gen_ai.operation.name", "")

        style = getattr(self.tracing_config, operation_name, None)

        if not style:
            return

        if span.is_llm_call():
            panels = []
            if messages := span.attributes.get("gen_ai.input.messages"):
                panels.append(
                    Panel(
                        JSON(messages), title="INPUT", style="white", title_align="left"
                    )
                )
            if output_panel := _get_output_panel(span):
                panels.append(output_panel)
            if usage := {
                k.replace("gen_ai.usage.", ""): v
                for k, v in span.attributes.items()
                if "usage" in k
            }:
                panels.append(
                    Panel(
                        JSON(json.dumps(usage)),
                        title="USAGE",
                        style="white",
                        title_align="left",
                    )
                )
            self.console.print(
                Panel(
                    Group(*panels),
                    title=f"{operation_name.upper()}: {span.attributes.get('gen_ai.request.model')}",
                    style=style,
                )
            )
        elif span.is_tool_execution():
            panels = [
                Panel(
                    JSON(span.attributes.get("gen_ai.tool.args", "{}")),
                    title="Input",
                    style="white",
                    title_align="left",
                )
            ]
            if output_panel := _get_output_panel(span):
                panels.append(output_panel)
            self.console.print(
                Panel(
                    Group(*panels),
                    title=f"{operation_name.upper()}: {span.attributes.get('gen_ai.tool.name')}",
                    style=style,
                )
            )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
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
                if (
                    self.tracing_config.cost_info
                    and span.attributes.get("gen_ai.operation.name") == "call_llm"
                ):
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
        trace_id = self.run_trace_mapping.pop(agent_run_id, None)
        if trace_id is None:
            msg = f"Trace ID not found for agent run ID: {agent_run_id}"
            raise ValueError(msg)
        trace = self.traces.pop(trace_id, None)
        if trace is None:
            msg = f"Trace not found for trace ID: {trace_id}"
            raise ValueError(msg)
        return trace
