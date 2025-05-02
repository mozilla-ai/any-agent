import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentFramework, AnyAgentSpan
from any_agent.telemetry import TelemetryProcessor


def test_telemetry_extract_interaction(
    agent_framework: AgentFramework, llm_span: ReadableSpan
) -> None:
    if agent_framework in (
        AgentFramework.AGNO,
        AgentFramework.GOOGLE,
        AgentFramework.TINYAGENT,
    ):
        pytest.skip()
    processor = TelemetryProcessor.create(AgentFramework(agent_framework))
    assert llm_span.attributes  # to make mypy happy
    span = AnyAgentSpan.from_readable_span(llm_span)
    span_kind, interaction = processor.extract_interaction(span)
    assert span_kind == "LLM"
    assert interaction["input"]
