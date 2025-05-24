from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentTrace
from any_agent.config import TracingConfig
from any_agent.tracing.exporter import _AnyAgentExporter


@pytest.fixture
def readable_spans(agent_trace: AgentTrace) -> list[ReadableSpan]:
    return [span.to_readable_span() for span in agent_trace.spans]


def test_rich_console_span_exporter_default(readable_spans: list[ReadableSpan]) -> None:
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = _AnyAgentExporter(TracingConfig())
        exporter.export(readable_spans)
        console_mock.return_value.print.assert_called()


def test_rich_console_span_exporter_disable(readable_spans: list[ReadableSpan]) -> None:
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = _AnyAgentExporter(TracingConfig(console=False))
        exporter.export(readable_spans)
        console_mock.return_value.print.assert_not_called()


def test_cost_info_span_exporter_disable(readable_spans: list[ReadableSpan]) -> None:
    add_cost_info = MagicMock()
    with patch("any_agent.tracing.exporter.AgentSpan.add_cost_info", add_cost_info):
        exporter = _AnyAgentExporter(TracingConfig(cost_info=False))
        exporter.export(readable_spans)
        add_cost_info.assert_not_called()
