from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from any_agent.tracing.trace import AgentTrace


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a criterion."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str
    criteria: str
    points: int


class AnswerDetails(TypedDict):
    answer_start: Sequence[int]
    text: Sequence[str]


class GroundTruthAnswers(TypedDict):
    id: str
    answers: AnswerDetails


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description."""

    model_config = ConfigDict(extra="forbid")
    criteria: str
    points: int


class GroundTruthAnswer(TypedDict):
    name: str
    value: float
    points: float


class TraceEvaluationResult(BaseModel):
    """Represents the result of evaluating a trace."""

    model_config = ConfigDict(extra="forbid")

    trace: AgentTrace
    hypothesis_answer: str | None
    checkpoint_results: list[EvaluationResult]
    hypothesis_answer_results: list[EvaluationResult]
    direct_results: list[EvaluationResult]

    @property
    def score(self) -> float:
        """Calculate the score based on the evaluation results."""
        all_results = (
            self.checkpoint_results
            + self.hypothesis_answer_results
            + self.direct_results
        )
        total_points = sum([result.points for result in all_results])
        if total_points == 0:
            msg = "Total points is 0, cannot calculate score."
            raise ValueError(msg)
        passed_points = sum([result.points for result in all_results if result.passed])
        return float(passed_points / total_points)
