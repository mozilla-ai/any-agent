from typing import TYPE_CHECKING, Any

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

from .any_agent import AnyAgent

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM
    from agno.team.team import Team

    DEFAULT_MODEL_TYPE = LiteLLM
    agno_available = True
except ImportError:
    agno_available = False


if TYPE_CHECKING:
    from agno.agent import RunResponse
    from agno.models.base import Model

    from any_agent.tracing.trace import AgentTrace


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
        self._agent: Agent | Team | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.AGNO

    def _get_model(self, agent_config: AgentConfig) -> "Model":
        """Get the model configuration for an Agno agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        return model_type(
            id=agent_config.model_id,
            api_base=agent_config.api_base,
            api_key=agent_config.api_key,
            **agent_config.model_args or {},
        )

    async def _load_agent(self) -> None:
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        if self.managed_agents:
            tools, _ = await self._load_tools(self.config.tools)

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
                members=members,  # type: ignore[arg-type]
                instructions=self.config.instructions,
                tools=tools,
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

    async def run_async(self, prompt: str, **kwargs: Any) -> "AgentTrace":
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        result: RunResponse = await self._agent.arun(prompt, **kwargs)
        self._exporter.trace.final_output = result.content
        return self._exporter.trace
