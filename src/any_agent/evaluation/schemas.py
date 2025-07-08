from pydantic import BaseModel


class EvaluationOutput(BaseModel):
    passed: bool
    """Whether the evaluation passed or failed."""

    reasoning: str
    """The reasoning for the evaluation."""
