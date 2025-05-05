from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing.exporter import AnyAgentExporter
from any_agent.tracing.trace import AgentTrace, is_tracing_supported


def test_exporter_initialization(
    agent_framework: AgentFramework, tmp_path: Path
) -> None:
    exporter = AnyAgentExporter(
        agent_framework=agent_framework,
        tracing_config=TracingConfig(
            output_dir=str(tmp_path / "traces"),
        ),
    )

    assert (tmp_path / "traces").exists()
    assert exporter.console is not None


def test_rich_console_span_exporter_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(AgentFramework.LANGCHAIN, TracingConfig(save=False))
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_called()


def test_rich_console_span_exporter_disable(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(save=False, llm=None),
        )
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_not_called()


def test_save_default(tmp_path, llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    exporter = AnyAgentExporter(
        AgentFramework.LANGCHAIN,
        TracingConfig(output_dir=str(tmp_path), console=False, save=True),
    )
    exporter.export([llm_span])
    # Just to simulate more than 1 span
    exporter.export([llm_span])
    trace_files = [str(x) for x in tmp_path.iterdir()]
    assert len(trace_files) == 1
    assert exporter.trace.output_file in trace_files


def test_cost_info_default(llm_span: ReadableSpan):  # type: ignore[no-untyped-def]
    console_mock = MagicMock()
    with patch("any_agent.tracing.exporter.Console", console_mock):
        exporter = AnyAgentExporter(
            AgentFramework.LANGCHAIN,
            TracingConfig(console=False, save=False),
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
            TracingConfig(save=False, console=False, cost_info=False),
        )
        exporter.export([llm_span])
        attributes = exporter.trace.spans[0].attributes
        for key in (
            "cost_prompt",
            "cost_completion",
        ):
            assert key not in attributes


def test_save_trace(
    agent_framework: AgentFramework, llm_span: ReadableSpan, tmp_path: Path
) -> None:
    if not is_tracing_supported(agent_framework):
        pytest.skip(
            f"Tracing is not supported for {agent_framework.name}. Skipping test."
        )
    #  This test assumes that these attributes are set in the span when the test starts
    assert llm_span.attributes
    assert llm_span.attributes.get("cost_prompt") is None
    assert llm_span.attributes.get("cost_completion") is None

    exporter = AnyAgentExporter(
        agent_framework=AgentFramework.LANGCHAIN,
        tracing_config=TracingConfig(
            output_dir=str(tmp_path), console=False, save=True, cost_info=True
        ),
    )
    exporter.export([llm_span])
    exporter.export([llm_span])
    assert exporter.trace.output_file
    # make sure that we can load it back
    with open(exporter.trace.output_file, encoding="utf-8") as f:
        trace_str = f.read()
    trace = AgentTrace.model_validate_json(trace_str)
    assert len(trace.spans) == 2
    assert trace.spans == exporter.trace.spans
    assert trace.output_file == exporter.trace.output_file
    assert trace.final_output is None
