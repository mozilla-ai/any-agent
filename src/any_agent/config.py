from collections.abc import Callable, Mapping, MutableMapping, Sequence
from enum import Enum, auto
from typing import Any, Self

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


class MCPStdioParams(BaseModel):
    command: str
    args: Sequence[str]
    tools: Sequence[str] = Field(default_factory=list)


class MCPSseParams(BaseModel):
    url: str
    headers: Mapping[str, str] = Field(default_factory=dict)
    tools: Sequence[str] = Field(default_factory=list)

MCPParams = MCPStdioParams | MCPSseParams

Tool = str | MCPParams | Callable[..., Any]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    description: str = ""
    name: str = "any_agent"
    instructions: str | None = None
    tools: Sequence[Tool] = Field(default_factory=list)
    handoff: bool = False
    agent_type: str | None = None
    agent_args: MutableMapping[str, Any] = Field(default_factory=dict)
    model_type: str | None = None
    model_args: MutableMapping[str, Any] = Field(default_factory=dict)


class TracingConfig(BaseModel):
    llm: str | None = "yellow"
    tool: str | None = "blue"
    agent: str | None = None
    chain: str | None = None
