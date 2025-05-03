from collections.abc import Callable, Mapping, MutableMapping, Sequence
from enum import Enum, auto
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentFramework(str, Enum):
    GOOGLE = auto()
    LANGCHAIN = auto()
    LLAMA_INDEX = auto()
    OPENAI = auto()
    AGNO = auto()
    SMOLAGENTS = auto()
    TINYAGENT = auto()

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

    model_config = ConfigDict(frozen=True, extra="forbid")


class MCPSseParams(BaseModel):
    url: str
    headers: Mapping[str, str] | None = None
    tools: Sequence[str] | None = None
    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class TracingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    console: bool = True
    """Print to console."""

    output_dir: str = "traces"
    """Directory to save traces, if json is enabled"""

    save: bool = True
    """Save to json file."""

    cost_info: bool = True
    """whether json and console logs should include cost information"""

    llm: str | None = "yellow"
    """LLM color in console logs"""

    tool: str | None = "blue"
    """Tool color in console logs"""

    agent: str | None = None
    """Agent color in console logs"""

    chain: str | None = None
    """Chain color in console logs"""

    @model_validator(mode="after")
    def validate_enable_flags(self) -> Self:
        if not self.console and not self.save:
            msg = "At least one of `console` or `save` must be true"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_output_dir(self) -> Self:
        if self.save and not self.output_dir:
            msg = "output_dir must be set if json is enabled"
            raise ValueError(msg)
        return self


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
    agent_type: Callable[..., Any] | None = None
    agent_args: MutableMapping[str, Any] | None = None
    model_type: Callable[..., Any] | None = None
    model_args: MutableMapping[str, Any] | None = None
