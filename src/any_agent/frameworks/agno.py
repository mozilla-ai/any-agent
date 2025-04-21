from typing import Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import search_web, visit_webpage

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM

    agno_available = True
except ImportError:
    agno_available = False


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.AGNO

    def _get_model(self, agent_config: AgentConfig) -> LiteLLM:
        """Get the model configuration for an Agno agent."""
        return LiteLLM(
            id=agent_config.model_id,
            **agent_config.model_args or {},
        )

    async def load_agent(self) -> None:
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)
        if self.managed_agents:
            msg = "Managed agents are not yet supported in Agno agent."
            raise NotImplementedError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]
        tools, _ = await self._load_tools(self.config.tools)

        self._agent: Agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions or "",
            model=self._get_model(self.config),
            tools=tools,
            **self.config.agent_args or {},
        )

    async def run_async(self, prompt: str) -> Any:
        return await self._agent.arun(prompt)
