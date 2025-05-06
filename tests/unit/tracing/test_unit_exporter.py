from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import ReadableSpan

from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing.exporter import AnyAgentExporter


def test_exporter_initialization(agent_framework: AgentFramework) -> None:
    exporter = AnyAgentExporter(
        agent_framework=agent_framework,
        tracing_config=TracingConfig(),
    )

    assert exporter.console is not None


def test_rich_console_span_exporter_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(AgentFramework.LANGCHAIN, TracingConfig())
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_called()


def test_rich_console_span_exporter_disable(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(llm=None),
        )
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_not_called()


def test_cost_info_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(
                console=False,
            ),
        )
        exporter.export([llm_span])
        attributes = exporter.trace.spans[0].attributes
        for key in (
            "cost_prompt",
            "cost_completion",
        ):
            assert key in attributes


def test_rich_console_cost_info_disabled(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(console=False, cost_info=False),
        )
        exporter.export([llm_span])
        attributes = exporter.trace.spans[0].attributes
        for key in (
            "cost_prompt",
            "cost_completion",
        ):
            assert key not in attributes
