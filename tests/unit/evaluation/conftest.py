import json

import pytest

from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.tracing.trace import AgentSpan, AgentTrace


@pytest.fixture
def evaluation_case():
    return EvaluationCase(
        ground_truth=[{"name": "Test Case 1", "value": 1.0, "points": 1.0}],
        checkpoints=[{"criteria": "Check if value is 1.0", "points": 1}],
        llm_judge="gpt-4o-mini",
        final_answer_criteria=[]
    )

@pytest.fixture
def agent_trace():
    trace_path = "tests/unit/evaluation/sample_traces/OPENAI.json"
    with open(trace_path, encoding="utf-8") as f:
        spans = json.loads(f.read())
    spans = [AgentSpan.model_validate_json(span) for span in spans]
    return AgentTrace(spans=spans)
