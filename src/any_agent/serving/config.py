from a2a.types import AgentSkill
from pydantic import BaseModel, ConfigDict


class A2AServingConfig(BaseModel):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A).

    Example:
        config = A2AServingConfig(
            port=8080,
            endpoint="/my-agent",
            skills=[
                AgentSkill(
                    id="search",
                    name="web_search",
                    description="Search the web for information"
                )
            ]
        )

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

    skills: list[AgentSkill] | None = None
    """List of skills to be used by the agent.

    If not provided, the skills will be inferred from the tools.
    """

    version: str = "0.1.0"
