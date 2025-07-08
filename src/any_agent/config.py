from collections.abc import Callable, Mapping, MutableMapping, Sequence
from enum import StrEnum, auto
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from any_agent.callbacks import get_default_callbacks
from any_agent.callbacks.base import Callback


class AgentFramework(StrEnum):
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


class MCPStdio(BaseModel):
    command: str
    """The executable to run to start the server.

    For example, `docker`, `uvx`, `npx`.
    """

    args: Sequence[str]
    """Command line args to pass to the command executable.

    For example, `["run", "-i", "--rm", "mcp/fetch"]`.
    """

    env: dict[str, str] | None = None
    """The environment variables to set for the server."""

    tools: Sequence[str] | None = None
    """List of tool names to use from the MCP Server.

    Use it to limit the tools accessible by the agent.
    For example, if you use [`mcp/filesystem`](https://hub.docker.com/r/mcp/filesystem),
    you can pass `tools=["read_file", "list_directory"]` to limit the agent to read-only operations.

    If none is specified, the default behavior is that the agent will have access to all tools under that MCP server.
    """

    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MCPSse(BaseModel):
    url: str
    """The URL of the server."""

    headers: Mapping[str, str] | None = None
    """The headers to send to the server."""

    tools: Sequence[str] | None = None
    """List of tool names to use from the MCP Server.

    Use it to limit the tools accessible by the agent.
    For example, if you use [`mcp/filesystem`](https://hub.docker.com/r/mcp/filesystem),
    you can pass `tools=["read_file", "list_directory"]` to limit the agent to read-only operations.
    """

    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class ServingConfig(BaseModel):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A).

    We use the example `A2ASever` from https://github.com/google/A2A/tree/main/samples/python.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    """Will be passed as argument to `uvicorn.run`."""

    port: int = 5000
    """Will be passed as argument to `uvicorn.run`."""

    endpoint: str = "/"
    """Will be pass as argument to `Starlette().add_route`"""

    log_level: str = "warning"
    """Will be passed as argument to the `uvicorn` server."""

    version: str = "0.1.0"


MCPParams = MCPStdio | MCPSse

Tool = str | MCPParams | Callable[..., Any]


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model_id: str
    """Select the underlying model used by the agent.

    If you are using the default model_type (LiteLLM), you can refer to [LiteLLM Provider Docs](https://docs.litellm.ai/docs/providers) for the list of providers and how to access them.
    """

    api_base: str | None = None
    api_key: str | None = None

    description: str | None = None
    """Description of the agent."""

    name: str = "any_agent"
    """The name of the agent.

    Defaults to `any_agent`.
    """

    instructions: str | None = None
    """Specify the instructions for the agent (often also referred to as a `system_prompt`)."""

    tools: Sequence[Tool] = Field(default_factory=list)
    """List of tools to be used by the agent.

    See more info at [Tools](../agents/tools.md).
    """

    callbacks: list[Callback] = Field(default_factory=get_default_callbacks)
    """List of callbacks to use during agent invocation.

    See more info at [Callbacks](../agents/callbacks.md).
    """

    agent_type: Callable[..., Any] | None = None
    """Control the type of agent class that is used by the framework, and is unique to the framework used.

    Check the individual `Frameworks` pages for more info on the defaults.
    """

    agent_args: MutableMapping[str, Any] | None = None
    """Pass arguments to the instance used by the underlying framework.

    For example, you can pass `output_type` when using the OpenAI Agents SDK:

    ```py
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    agent = AnyAgent.create(
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Extract calendar events from text",
            agent_args={
                "output_type": CalendarEvent
            }
        )
    )
    ```
    """

    model_type: Callable[..., Any] | None = None
    """Control the type of model class that is used by the agent framework, and is unique to the agent framework being used.

    For each framework, we leverage their support for LiteLLM and use it as default model_type, allowing you to use the same model_id syntax across these frameworks.
    """

    model_args: MutableMapping[str, Any] | None = None
    """Pass arguments to the model instance like `temperature`, `top_k`, as well as any other provider-specific parameters.

    Refer to LiteLLM Completion API Docs for more info.
    """

    output_type: type[BaseModel] | None = None
    """Control the output schema from calling `run`. By default, the agent will return a type str.

    Using this parameter you can define a Pydantic model that will be returned by the agent run methods.
    """
