from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


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
