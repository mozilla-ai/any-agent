import json

import pytest

from any_agent import AgentFramework
from any_agent.telemetry import TelemetryProcessor


def test_telemetry_extract_interaction(agent_framework: AgentFramework, llm_span):  # type: ignore[no-untyped-def]
    if agent_framework in (AgentFramework.AGNO, AgentFramework.GOOGLE):
        pytest.skip()
    processor = TelemetryProcessor.create(AgentFramework(agent_framework))
    span_kind, interaction = processor.extract_interaction(
        json.loads(llm_span.to_json())
    )
    assert span_kind == "LLM"
    assert interaction["input"]
