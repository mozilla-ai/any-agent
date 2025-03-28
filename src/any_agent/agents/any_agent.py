from typing import Any, Optional
from abc import ABC, abstractmethod

from any_agent.schema import AgentFramework, AgentSchema


class AnyAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    # factory method
    @classmethod
    def create(
        cls,
        agent_framework: AgentFramework,
        agent_config: AgentSchema,
        managed_agents: Optional[list[AgentSchema]] = None,
    ) -> "AnyAgent":
        # Import here to avoid circular imports
        from any_agent.agents.langchain_agent import LangchainAgent
        from any_agent.agents.openai_agent import OpenAIAgent
        from any_agent.agents.smolagents_agent import SmolagentsAgent

        if agent_framework == AgentFramework.SMOLAGENTS:
            return SmolagentsAgent(agent_config, managed_agents=managed_agents)
        elif agent_framework == AgentFramework.LANGCHAIN:
            return LangchainAgent(agent_config, managed_agents=managed_agents)
        elif agent_framework == AgentFramework.OPENAI:
            return OpenAIAgent(agent_config, managed_agents=managed_agents)
        else:
            raise ValueError(f"Unsupported agent framework: {agent_framework}")

    @abstractmethod
    def _load_agent(self) -> None:
        """Load the agent instance."""
        pass

    @abstractmethod
    def run(self, prompt: str) -> Any:
        """Run the agent with the given prompt."""
        pass
