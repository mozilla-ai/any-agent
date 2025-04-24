from typing import TYPE_CHECKING, Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import search_web, visit_webpage

try:
    import smolagents

    smolagents_available = True
except ImportError:
    smolagents_available = False

if TYPE_CHECKING:
    from smolagents import MultiStepAgent


DEFAULT_AGENT_TYPE = "CodeAgent"
DEFAULT_MODEL_CLASS = "LiteLLMModel"


class SmolagentsAgent(AnyAgent):
    """Smolagents agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
    ):
        super().__init__(config, managed_agents)
        self._agent: MultiStepAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def _get_model(self, agent_config: AgentConfig) -> Any:
        """Get the model configuration for a smolagents agent."""
        model_type = getattr(smolagents, agent_config.model_type or DEFAULT_MODEL_CLASS)
        kwargs = {
            "model_id": agent_config.model_id,
            "api_key": agent_config.api_key,
            "api_base": agent_config.api_base,
        }
        model_args = agent_config.model_args or {}
        return model_type(**kwargs, **model_args)

    async def load_agent(self) -> None:
        """Load the Smolagents agent with the given configuration."""
        if not smolagents_available:
            msg = "You need to `pip install 'any-agent[smolagents]'` to use this agent"
            raise ImportError(msg)
        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        tools, _ = await self._load_tools(self.config.tools)

        managed_agents_instanced = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                agent_type = getattr(
                    smolagents,
                    managed_agent.agent_type or DEFAULT_AGENT_TYPE,
                )
                managed_tools, _ = await self._load_tools(managed_agent.tools)
                managed_agent_instance = agent_type(
                    name=managed_agent.name,
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    verbosity_level=-1,  # OFF
                    description=managed_agent.description
                    or f"Use the agent: {managed_agent.name}",
                )
                if managed_agent.instructions:
                    managed_agent_instance.prompt_templates["system_prompt"] = (
                        managed_agent.instructions
                    )
                managed_agents_instanced.append(managed_agent_instance)

        main_agent_type = getattr(
            smolagents,
            self.config.agent_type or DEFAULT_AGENT_TYPE,
        )

        self._agent = main_agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=tools,
            verbosity_level=-1,  # OFF
            managed_agents=managed_agents_instanced,
            **self.config.agent_args or {},
        )

        assert self._agent

        if self.config.instructions:
            self._agent.prompt_templates["system_prompt"] = self.config.instructions

    async def run_async(self, prompt: str) -> Any:
        """Run the Smolagents agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)

        return self._agent.run(prompt)
