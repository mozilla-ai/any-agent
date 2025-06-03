from typing import Literal

from a2a.types import AgentSkill

from any_agent.config import ServingConfig


class A2AServingConfig(ServingConfig):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A).

    Example:
        >>> config = A2AServingConfig(
        ...     port=8080,
        ...     endpoint="/my-agent",
        ...     skills=[
        ...         AgentSkill(
        ...             id="search",
        ...             name="web_search",
        ...             description="Search the web for information"
        ...         )
        ...     ]
        ... )

    """

    skills: list[AgentSkill] | None = None
    """List of skills to be used by the agent.

    If not provided, the skills will be inferred from the tools.
    """

    type: Literal["a2a"] = "a2a"
