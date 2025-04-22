from typing import Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM
    from agno.team.team import Team

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
            api_base=agent_config.api_base,
            api_key=agent_config.api_key,
            **agent_config.model_args or {},
        )

    async def load_agent(self) -> None:
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        self._agent: Agent | Team

        if self.managed_agents:
            if self.config.tools:
                msg = "The main agent can't use tools in agno."
                raise ValueError(msg)

            members = []
            for n, managed_agent in enumerate(self.managed_agents):
                managed_tools, _ = await self._load_tools(managed_agent.tools)
                name = managed_agent.name
                if not name or name == "any_agent":
                    logger.warning(
                        "Overriding name for managed_agent. Can't use the default.",
                    )
                    name = f"managed_agent_{n}"
                members.append(
                    Agent(
                        name=name,
                        role=managed_agent.description,
                        instructions=managed_agent.instructions,
                        model=self._get_model(managed_agent),
                        tools=managed_tools,
                        **managed_agent.agent_args or {},
                    )
                )

            self._agent = Team(
                mode="collaborate",
                name=f"Team managed by agent {self.config.name}",
                description=self.config.description,
                model=self._get_model(self.config),
                members=members,
                instructions=self.config.instructions,
                **self.config.agent_args or {},
            )
        else:
            tools, _ = await self._load_tools(self.config.tools)

            self._agent = Agent(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self._get_model(self.config),
                tools=tools,
                **self.config.agent_args or {},
            )

    async def run_async(self, prompt: str) -> Any:
        return await self._agent.arun(prompt)
