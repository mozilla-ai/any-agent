from unittest.mock import MagicMock, patch

from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing import RichConsoleSpanExporter


def test_rich_console_span_exporter_default(llm_span):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(AgentFramework.LANGCHAIN, TracingConfig())
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_called()


def test_rich_console_span_exporter_disable(llm_span):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(llm=None),
        )
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_not_called()
