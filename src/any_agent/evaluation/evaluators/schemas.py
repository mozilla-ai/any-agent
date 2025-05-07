from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a criterion."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str
    criteria: str
    points: int
