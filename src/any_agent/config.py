from collections.abc import Callable
from enum import Enum, auto
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class AgentFramework(str, Enum):
    GOOGLE = auto()
    LANGCHAIN = auto()
    LLAMA_INDEX = auto()
    OPENAI = auto()
    AGNO = auto()
    SMOLAGENTS = auto()

    @classmethod
    def from_string(cls, value: str | Self) -> Self:
        if isinstance(value, cls):
            return value

        formatted_value = value.strip().upper()
        if formatted_value not in cls.__members__:
            error_message = (
                f"Unsupported agent framework: '{value}'. "
                f"Valid frameworks are: {list(cls.__members__.keys())}"
            )
            raise ValueError(error_message)

        return cls[formatted_value]


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
    tools: list[str | MCPTool | Callable] = Field(default_factory=list)
    handoff: bool = False
    agent_type: str | None = None
    agent_args: dict | None = None
    model_type: str | None = None
    model_args: dict | None = None
    description: str | None = None
