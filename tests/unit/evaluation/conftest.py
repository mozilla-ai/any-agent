import pytest

from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.schemas import CheckpointCriteria


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
