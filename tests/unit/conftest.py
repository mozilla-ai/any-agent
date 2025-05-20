import json
from pathlib import Path
import pytest

from any_agent.tracing.trace import AgentTrace


@pytest.fixture
def agent_trace() -> AgentTrace:
    trace_path = Path(__file__).parent.parent / "assets" / "OPENAI.json"
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    return AgentTrace.model_validate(trace)
