from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import ReadableSpan

from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing import RichConsoleSpanExporter


def test_rich_console_span_exporter_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(AgentFramework.LANGCHAIN, TracingConfig())
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_called()


def test_rich_console_span_exporter_disable(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(llm=None),
        )
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_not_called()


def test_rich_console_cost_info_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(),
        )
        exporter.export([llm_span])
        print_args = [
            args[0][0] for args in console_mock.return_value.print.call_args_list
        ]
        for key in (
            "token_count_prompt",
            "token_count_completion",
            "cost_prompt",
            "cost_completion",
        ):
            assert any(key in arg for arg in print_args)


def test_rich_console_cost_info_disabled(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(cost_info=False),
        )
        exporter.export([llm_span])
        print_args = [
            args[0][0] for args in console_mock.return_value.print.call_args_list
        ]
        for key in (
            "token_count_prompt",
            "token_count_completion",
            "cost_prompt",
            "cost_completion",
        ):
            assert not any(key in arg for arg in print_args)
