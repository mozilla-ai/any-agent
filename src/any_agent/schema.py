from pydantic import BaseModel, Field


class AgentSchema(BaseModel):
    model_id: str
    name: str = "default-name"
    instructions: str | None = None
    api_base: str | None = None
    api_key_var: str | None = None
    tools: list[str] = Field(default_factory=list)
    handoff: bool = False
    agent_class: str | None = None
    model_class: str | None = None
    description: str | None = None
