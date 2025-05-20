import json
from pathlib import Path

import pytest

from any_agent.tracing.trace import AgentTrace


@pytest.fixture(
    params=list((Path(__file__).parent.parent / "assets").glob("*_trace.json")),
    ids=lambda x: Path(x).stem,
)
def agent_trace(request: pytest.FixtureRequest) -> AgentTrace:
    trace_path = request.param
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    return AgentTrace.model_validate(trace)
