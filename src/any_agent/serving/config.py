from typing import Literal

from a2a.types import AgentSkill

from any_agent.config import ServingConfig


class A2AServingConfig(ServingConfig):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A)."""

    type: Literal["a2a"] = "a2a"
    """Type discriminator for A2A serving configuration."""

    skills: list["AgentSkill"] | None = None
    """List of skills to be used by the agent.

    If not provided, the skills will be inferred from the tools.
    """

    def get_config_type(self) -> str:
        """Implementation of abstract method from ServingConfig."""
        return self.type
