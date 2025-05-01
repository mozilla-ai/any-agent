import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentFramework, AnyAgentSpan, TelemetryProcessor


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
    span = AnyAgentSpan(
        name=llm_span.name,
        kind=llm_span.kind,
        context=llm_span.context,
        parent=llm_span.parent,
        start_time=llm_span.start_time,
        end_time=llm_span.end_time,
        events=llm_span.events,
        attributes=dict(
            llm_span.attributes
        ),  # turn the mapping into a dict so that it's mutable
        status=llm_span.status,
        links=llm_span.links,
        resource=llm_span.resource,
    )
    span_kind, interaction = processor.extract_interaction(span)
    assert span_kind == "LLM"
    assert interaction["input"]
