from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class AgentFramework(str, Enum):
    GOOGLE = "google"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llama_index"
    OPENAI = "openai"
    AGNO = "agno"
    SMOLAGENTS = "smolagents"


class MCPTool(BaseModel):
    command: str
    args: list[str]
    tools: list[str] | None = None


class TracingConfig(BaseModel):
    llm: str | None = "yellow"
    tool: str | None = "blue"
    agent: str | None = None
    chain: str | None = None


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    name: str = "any_agent"
    instructions: str | None = None
    tools: list[str | MCPTool] = Field(default_factory=list)
    handoff: bool = False
    agent_type: str | None = None
    agent_args: dict | None = None
    model_type: str | None = None
    model_args: dict | None = None
    description: str | None = None
