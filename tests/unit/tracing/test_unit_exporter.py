from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentTrace
from any_agent.config import TracingConfig
from any_agent.tracing.exporter import (
    _AnyAgentConsoleExporter,
    _AnyAgentExporter,
    _get_output_panel,
)


@pytest.fixture
def readable_spans(agent_trace: AgentTrace) -> list[ReadableSpan]:
    return [span.to_readable_span() for span in agent_trace.spans]


def test_console_exporter_default(
    agent_trace: AgentTrace, request: pytest.FixtureRequest
) -> None:
    """Test that the console exporter prints to console when enabled."""
    console_mock = MagicMock()
    panel_mock = MagicMock()
    markdown_mock = MagicMock()
    readable_spans = [span.to_readable_span() for span in agent_trace.spans]
    with (
        patch("any_agent.tracing.exporter.Console", console_mock),
        patch("any_agent.tracing.exporter.Markdown", markdown_mock),
        patch("any_agent.tracing.exporter.Panel", panel_mock),
    ):
        exporter = _AnyAgentConsoleExporter(TracingConfig(console=True))
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


def test_console_exporter_disabled() -> None:
    """Test that console exporter with console=False doesn't initialize console."""
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = _AnyAgentConsoleExporter(TracingConfig(console=False))
        # Console should not be initialized when console=False
        assert exporter.console is None
        console_mock.assert_not_called()


def test_trace_exporter_no_console_output(readable_spans: list[ReadableSpan]) -> None:
    """Test that the main trace exporter doesn't handle console output."""
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = _AnyAgentExporter(TracingConfig(console=True))
        exporter.export(readable_spans)
        # The main exporter should not print to console even if console=True
        console_mock.assert_not_called()


def test_trace_exporter_collects_traces(readable_spans: list[ReadableSpan]) -> None:
    """Test that the main trace exporter properly collects traces."""
    exporter = _AnyAgentExporter(TracingConfig())

    # Should start with empty traces
    assert len(exporter.traces) == 0
    assert len(exporter.run_trace_mapping) == 0

    # Export spans
    exporter.export(readable_spans)

    # Should have collected traces
    assert len(exporter.traces) > 0


def test_cost_info_span_exporter_disable(readable_spans: list[ReadableSpan]) -> None:
    """Test that cost info can be disabled."""
    add_cost_info = MagicMock()
    with patch("any_agent.tracing.exporter.AgentSpan.add_cost_info", add_cost_info):
        exporter = _AnyAgentExporter(TracingConfig(cost_info=False))
        exporter.export(readable_spans)
        add_cost_info.assert_not_called()


def test_cost_info_span_exporter_enabled(readable_spans: list[ReadableSpan]) -> None:
    """Test that cost info is added when enabled."""
    add_cost_info = MagicMock()
    with patch("any_agent.tracing.exporter.AgentSpan.add_cost_info", add_cost_info):
        exporter = _AnyAgentExporter(TracingConfig(cost_info=True))
        exporter.export(readable_spans)
        # Should be called for LLM spans
        add_cost_info.assert_called()


def test_get_output_panel(
    readable_spans: list[ReadableSpan], request: pytest.FixtureRequest
) -> None:
    """Test output panel generation for different span types."""
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
