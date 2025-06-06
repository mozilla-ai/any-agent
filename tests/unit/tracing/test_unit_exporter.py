from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentTrace
from any_agent.tracing.exporter import _ConsoleExporter, _get_output_panel


@pytest.fixture
def readable_spans(agent_trace: AgentTrace) -> list[ReadableSpan]:
    return [span.to_readable_span() for span in agent_trace.spans]


def test_rich_console_span_exporter_default(
    agent_trace: AgentTrace, request: pytest.FixtureRequest
) -> None:
    console_mock = MagicMock()
    panel_mock = MagicMock()
    markdown_mock = MagicMock()
    readable_spans = [span.to_readable_span() for span in agent_trace.spans]
    with (
        patch("any_agent.tracing.exporter.Console", console_mock),
        patch("any_agent.tracing.exporter.Markdown", markdown_mock),
        patch("any_agent.tracing.exporter.Panel", panel_mock),
    ):
        exporter = _ConsoleExporter()
        exporter.export(readable_spans)
        console_mock.return_value.print.assert_called()
        # TINYAGENT ends with a `task_completed` tool call
        if request.node.callspec.id not in ("TINYAGENT_trace",):
            panel_mock.assert_any_call(
                markdown_mock(agent_trace.final_output),
                title="OUTPUT",
                style="white",
                title_align="left",
            )


def test_get_output_panel(
    readable_spans: list[ReadableSpan], request: pytest.FixtureRequest
) -> None:
    # First LLM call returns JSON
    panel_mock = MagicMock()
    json_mock = MagicMock()
    with (
        patch("any_agent.tracing.exporter.Panel", panel_mock),
        patch("any_agent.tracing.exporter.JSON", json_mock),
    ):
        _get_output_panel(readable_spans[0])
        json_mock.assert_called_once()
        panel_mock.assert_called_once()

    if request.node.callspec.id not in ("LLAMA_INDEX_trace",):
        # First TOOL execution returns JSON
        panel_mock = MagicMock()
        json_mock = MagicMock()
        with (
            patch("any_agent.tracing.exporter.Panel", panel_mock),
            patch("any_agent.tracing.exporter.JSON", json_mock),
        ):
            _get_output_panel(readable_spans[1])
            json_mock.assert_called_once()
            panel_mock.assert_called_once()

    if request.node.callspec.id not in ("TINYAGENT_trace",):
        # Final LLM call returns string
        panel_mock = MagicMock()
        json_mock = MagicMock()
        with (
            patch("any_agent.tracing.exporter.Panel", panel_mock),
            patch("any_agent.tracing.exporter.JSON", json_mock),
        ):
            _get_output_panel(readable_spans[-2])
            json_mock.assert_not_called()
            panel_mock.assert_called_once()

    # AGENT invocation has no output
    panel_mock = MagicMock()
    json_mock = MagicMock()
    with (
        patch("any_agent.tracing.exporter.Panel", panel_mock),
        patch("any_agent.tracing.exporter.JSON", json_mock),
    ):
        _get_output_panel(readable_spans[-1])
        json_mock.assert_not_called()
        panel_mock.assert_not_called()
