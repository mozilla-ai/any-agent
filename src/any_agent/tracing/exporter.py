# mypy: disable-error-code="arg-type,union-attr"
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan


SCOPE_NAME = "any_agent"


def _get_output_panel(span: ReadableSpan) -> Panel | None:
    if output := span.attributes.get("gen_ai.output", None):
        output_type = span.attributes.get("gen_ai.output.type", "text")
        return Panel(
            Markdown(output) if output_type != "json" else JSON(output),
            title="OUTPUT",
            style="white",
            title_align="left",
        )
    return None


class _ConsoleExporter(SpanExporter):
    def __init__(self) -> None:
        self.console: Console = Console()

    def print_to_console(self, span: ReadableSpan) -> None:
        if scope := span.instrumentation_scope:
            if scope.name != SCOPE_NAME:
                return

        if not span.attributes:
            return

        operation_name = span.attributes.get("gen_ai.operation.name", "")

        if operation_name == "call_llm":
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
                    style="yellow",
                )
            )
        elif operation_name == "execute_tool":
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
                    style="blue",
                )
            )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for readable_span in spans:
            self.print_to_console(readable_span)
        return SpanExportResult.SUCCESS
