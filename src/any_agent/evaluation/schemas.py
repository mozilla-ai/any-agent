from pydantic import BaseModel


class EvaluationOutput(BaseModel):
    passed: bool
    reasoning: str
