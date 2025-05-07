from typing import TYPE_CHECKING, Any

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.tools import search_web, visit_webpage

from .any_agent import AnyAgent

try:
    from agents import (
        Agent,
        Handoff,
        Model,
        ModelSettings,
        Runner,
    )
    from agents.extensions.models.litellm_model import LitellmModel

    DEFAULT_MODEL_TYPE = LitellmModel

    agents_available = True
except ImportError:
    agents_available = False


if TYPE_CHECKING:
    from agents import Model

    from any_agent.tracing.trace import AgentTrace


class OpenAIAgent(AnyAgent):
    """OpenAI agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
        self._agent: Agent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.OPENAI

    def _get_model(
        self,
        agent_config: AgentConfig,
    ) -> "Model":
        """Get the model configuration for an OpenAI agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        return model_type(
            model=agent_config.model_id,
            base_url=agent_config.api_base,
            api_key=agent_config.api_key,
        )

    async def _load_agent(self) -> None:
        """Load the OpenAI agent with the given configuration."""
        if not agents_available:
            msg = "You need to `pip install 'any-agent[openai]'` to use this agent"
            raise ImportError(msg)
        if not agents_available:
            msg = "You need to `pip install openai-agents` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]
        tools, mcp_servers = await self._load_tools(self.config.tools)
        tools = self._filter_mcp_tools(tools, mcp_servers)

        handoffs = list[Agent[Any] | Handoff[Any]]()
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, managed_mcp_servers = await self._load_tools(
                    managed_agent.tools
                )
                managed_tools = self._filter_mcp_tools(managed_tools, mcp_servers)
                managed_agent_args = managed_agent.agent_args or {}
                handoff = managed_agent_args.pop("handoff", None)
                if managed_agent.model_args:
                    managed_agent_args["model_settings"] = managed_agent.model_args
                instance = Agent(
                    name=managed_agent.name,
                    instructions=managed_agent.instructions,
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    mcp_servers=[
                        managed_mcp_server.server
                        for managed_mcp_server in managed_mcp_servers
                    ],
                    **managed_agent_args,
                )
                if handoff:
                    handoffs.append(instance)
                else:
                    tools.append(
                        instance.as_tool(
                            tool_name=instance.name,
                            tool_description=managed_agent.description
                            or f"Use the agent: {managed_agent.name}",
                        ),
                    )

        kwargs_ = self.config.agent_args or {}
        if self.config.model_args:
            kwargs_["model_settings"] = ModelSettings(**self.config.model_args)
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            handoffs=handoffs,
            tools=tools,
            mcp_servers=[mcp_server.server for mcp_server in mcp_servers],
            **kwargs_,
        )

    def _filter_mcp_tools(self, tools: list[Any], mcp_servers: list[Any]) -> list[Any]:
        """OpenAI frameowrk doesn't expect the mcp tool to be included in `tools`."""
        non_mcp_tools = []
        for tool in tools:
            if any(tool in mcp_server.tools for mcp_server in mcp_servers):
                continue
            non_mcp_tools.append(tool)
        return non_mcp_tools

    async def run_async(self, prompt: str, **kwargs: Any) -> "AgentTrace":
        """Run the OpenAI agent with the given prompt asynchronously."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        result = await Runner.run(self._agent, prompt, **kwargs)
        self._exporter.trace.final_output = result.final_output
        return self._exporter.trace
