from collections.abc import Callable, MutableMapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentFramework(str, Enum):
    GOOGLE = "google"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llama_index"
    OPENAI = "openai"
    AGNO = "agno"
    SMOLAGENTS = "smolagents"


class MCPTool(BaseModel):
    command: str
    args: Sequence[str]
    tools: Sequence[str] | None = None


class TracingConfig(BaseModel):
    llm: str | None = "yellow"
    tool: str | None = "blue"
    agent: str | None = None
    chain: str | None = None


Tool = str | MCPTool | Callable[..., Any]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    description: str = ""
    name: str = "any_agent"
    instructions: str | None = None
    tools: Sequence[Tool] = Field(default_factory=list)
    handoff: bool = False
    agent_type: str | None = None
    agent_args: MutableMapping[str, Any] | None = None
    model_type: str | None = None
    model_args: MutableMapping[str, Any] | None = None
