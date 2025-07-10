# mypy: disable-error-code="arg-type,attr-defined,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.callbacks.base import Callback
from any_agent.tracing import span_attrs

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from any_agent.callbacks.context import Context


def _get_output_panel(span: ReadableSpan) -> Panel | None:
    if output := span.attributes.get(span_attrs.OUTPUT, None):
        output_type = span.attributes.get(span_attrs.OUTPUT_TYPE, "text")
        return Panel(
            Markdown(output) if output_type != "json" else JSON(output),
            title="OUTPUT",
            style="white",
            title_align="left",
        )
    return None


class ConsolePrintSpan(Callback):
    """Use rich's console to print the `Context.current_span`."""

    def __init__(self, console: Console | None = None) -> None:
        """Init the ConsolePrintSpan.

        Args:
            console: An optional instance of `rich.console.Console`.
                If `None`, a new instance will be used.

        """
        self.console = console or Console()

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        span = context.current_span

        operation_name = span.attributes.get(span_attrs.OPERATION, "")

        if operation_name != "call_llm":
            return context

        panels = []

        if messages := span.attributes.get(span_attrs.INPUT_MESSAGES):
            panels.append(
                Panel(JSON(messages), title="INPUT", style="white", title_align="left")
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
                title=f"{operation_name.upper()}: {span.attributes.get(span_attrs.MODEL_ID)}",
                style="yellow",
            )
        )

        return context

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        span = context.current_span

        operation_name = span.attributes.get(span_attrs.OPERATION, "")

        if operation_name != "execute_tool":
            return context

        panels = [
            Panel(
                JSON(span.attributes.get(span_attrs.TOOL_ARGS, "{}")),
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
        return context
