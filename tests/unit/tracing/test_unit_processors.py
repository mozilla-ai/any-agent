import pytest
from opentelemetry.sdk.trace import ReadableSpan

from any_agent import AgentFramework
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentSpan


def test_extract_interaction(
    agent_framework: AgentFramework, llm_span: ReadableSpan
) -> None:
    if agent_framework in (
        AgentFramework.AGNO,
        AgentFramework.GOOGLE,
        AgentFramework.TINYAGENT,
    ):
        pytest.skip()
    processor = TracingProcessor.create(AgentFramework(agent_framework))
    # to make mypy happy
    assert processor
    assert llm_span.attributes

    span = AgentSpan.from_readable_span(llm_span)
    span_kind, interaction = processor.extract_interaction(span)
    assert span_kind == "LLM"
    assert interaction["input"]
