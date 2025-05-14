import json

import pytest

from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.schemas import CheckpointCriteria
from any_agent.tracing.trace import AgentSpan, AgentTrace


@pytest.fixture
def evaluation_case() -> EvaluationCase:
    return EvaluationCase(
        ground_truth={"value": 1.0, "points": 1.0},
        checkpoints=[
            CheckpointCriteria.model_validate(
                {"criteria": "Check if the agent ran a calculation", "points": 1}
            )
        ],
        llm_judge="gpt-4o-mini",
    )


@pytest.fixture
def agent_trace() -> AgentTrace:
    trace_path = "tests/unit/evaluation/sample_traces/OPENAI.json"
    with open(trace_path, encoding="utf-8") as f:
        spans = json.loads(f.read())
    spans = [AgentSpan.model_validate_json(span) for span in spans]
    return AgentTrace(spans=spans, final_output="Final output")
