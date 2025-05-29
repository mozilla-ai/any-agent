from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from any_agent.tracing.agent_trace import AgentTrace


class AgentOutput(BaseModel):
    passed: bool
    reasoning: str


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a criterion."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str
    criteria: str | Callable[[AgentTrace], AgentOutput]
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
    criteria: str | Callable[[AgentTrace], AgentOutput]
    points: int


class GroundTruthAnswer(TypedDict):
    value: float
    points: float


class TraceEvaluationResult(BaseModel):
    """Represents the result of evaluating a trace."""

    model_config = ConfigDict(extra="forbid")

    trace: AgentTrace
    checkpoint_results: list[EvaluationResult]
    ground_truth_result: EvaluationResult | None = None

    @property
    def score(self) -> float:
        """Calculate the score based on the evaluation results."""
        if self.ground_truth_result is not None:
            all_results = [*self.checkpoint_results, self.ground_truth_result]
        else:
            all_results = self.checkpoint_results
        total_points = sum([result.points for result in all_results])
        if total_points == 0:
            msg = "Total points is 0, cannot calculate score."
            raise ValueError(msg)
        passed_points = sum([result.points for result in all_results if result.passed])
        return float(passed_points / total_points)
