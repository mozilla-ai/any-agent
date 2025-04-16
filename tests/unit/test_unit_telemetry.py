import json

import pytest

from any_agent import AgentFramework
from any_agent.telemetry import TelemetryProcessor


@pytest.mark.parametrize("framework", list(AgentFramework))
def test_telemetry_extract_interaction(framework, llm_span):
    if framework in ("agno", "google"):
        pytest.skip()
    processor = TelemetryProcessor.create(AgentFramework(framework))
    span_kind, interaction = processor.extract_interaction(
        json.loads(llm_span.to_json())
    )
    assert span_kind == "LLM"
    assert interaction["input"]
