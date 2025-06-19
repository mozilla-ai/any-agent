from pydantic import BaseModel


class AgentOutput(BaseModel):
    passed: bool
    reasoning: str
