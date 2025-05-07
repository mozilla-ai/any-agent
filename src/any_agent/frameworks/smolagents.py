from typing import TYPE_CHECKING, Any

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import search_web, visit_webpage

try:
    from smolagents import LiteLLMModel, ToolCallingAgent

    smolagents_available = True
except ImportError:
    smolagents_available = False

if TYPE_CHECKING:
    from smolagents import MultiStepAgent

    from any_agent.tracing.trace import AgentTrace


DEFAULT_AGENT_TYPE = ToolCallingAgent
DEFAULT_MODEL_TYPE = LiteLLMModel


class SmolagentsAgent(AnyAgent):
    """Smolagents agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
        self._agent: MultiStepAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def _get_model(self, agent_config: AgentConfig) -> Any:
        """Get the model configuration for a smolagents agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        kwargs = {
            "model_id": agent_config.model_id,
            "api_key": agent_config.api_key,
            "api_base": agent_config.api_base,
            **model_args,
        }
        return model_type(**kwargs)

    async def _load_agent(self) -> None:
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
                agent_type = managed_agent.agent_type or DEFAULT_AGENT_TYPE
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

        main_agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE

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

    async def run_async(self, prompt: str, **kwargs: Any) -> "AgentTrace":
        """Run the Smolagents agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        result = self._agent.run(prompt, **kwargs)
        self._exporter.trace.final_output = result
        return self._exporter.trace
