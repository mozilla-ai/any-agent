from collections.abc import Callable, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from any_agent.tracing.agent_trace import AgentTrace


class AgentOutput(BaseModel):
    passed: bool
    reasoning: str


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a criterion."""

    model_config = ConfigDict(extra="forbid")
    id: str
    """The identifier for the result, corresponds to the id of the criteria."""

    passed: bool
    """Whether the criteria was passed."""

    reason: str
    """The reason for the result."""

    criteria: str | Callable[[AgentTrace], AgentOutput]
    """The criteria to evaluate the agent's output against."""

    points: int
    """The number of points the criteria is worth."""


class AnswerDetails(TypedDict):
    answer_start: Sequence[int]
    text: Sequence[str]


class GroundTruthAnswers(TypedDict):
    id: str
    answers: AnswerDetails


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    """The unique identifier for the criteria."""

    criteria: str | Callable[[AgentTrace], AgentOutput]
    """The criteria to evaluate the agent's output against."""

    points: int
    """The number of points the criteria is worth."""


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
