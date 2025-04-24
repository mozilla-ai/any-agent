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
    tools: Sequence[str] | None = None
    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class MCPSseParams(BaseModel):
    url: str
    headers: Mapping[str, str] | None = None
    tools: Sequence[str] | None = None
    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class TracingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str | None = "traces"  # None for no json saved trace
    llm: str | None = "yellow"
    tool: str | None = "blue"
    agent: str | None = None
    chain: str | None = None


MCPParams = MCPStdioParams | MCPSseParams

Tool = str | MCPParams | Callable[..., Any]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    api_base: str | None = None
    api_key: str | None = None
    description: str | None = None
    name: str = "any_agent"
    instructions: str | None = None
    tools: Sequence[Tool] = Field(default_factory=list)
    handoff: bool = False
    agent_type: str | None = None
    agent_args: MutableMapping[str, Any] | None = None
    model_type: str | None = None
    model_args: MutableMapping[str, Any] | None = None
