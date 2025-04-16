import os
from typing import Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import wrap_tools

try:
    import smolagents
    from smolagents import MultiStepAgent

    smolagents_available = True
except ImportError:
    smolagents_available = None

DEFAULT_AGENT_TYPE = "CodeAgent"
DEFAULT_MODEL_CLASS = "LiteLLMModel"


class SmolagentsAgent(AnyAgent):
    """Smolagents agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: list[AgentConfig] | None = None
    ):
        if not smolagents_available:
            msg = "You need to `pip install 'any-agent[smolagents]'` to use this agent"
            raise ImportError(msg)
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        self._mcp_servers = None
        self._managed_mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a smolagents agent."""
        model_type = getattr(smolagents, agent_config.model_type or DEFAULT_MODEL_CLASS)
        kwargs = {
            "model_id": agent_config.model_id,
        }
        model_args = agent_config.model_args or {}
        if api_key_var := model_args.pop("api_key_var", None):
            kwargs["api_key"] = os.environ[api_key_var]
        return model_type(**kwargs, **model_args)

    def _merge_mcp_tools(self, mcp_servers):
        """Merge MCP tools from different servers."""
        tools = []
        for mcp_server in mcp_servers:
            tools.extend(mcp_server.tools)
        return tools

    async def _load_agent(self) -> None:
        """Load the Smolagents agent with the given configuration."""

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        tools, mcp_servers = await wrap_tools(
            self.config.tools, agent_framework=AgentFramework.SMOLAGENTS
        )
        self._mcp_servers = mcp_servers
        tools.extend(self._merge_mcp_tools(mcp_servers))

        managed_agents_instanced = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                agent_type = getattr(
                    smolagents, managed_agent.agent_type or DEFAULT_AGENT_TYPE
                )
                managed_tools, managed_mcp_servers = await wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.SMOLAGENTS
                )
                self._managed_mcp_servers = managed_mcp_servers
                tools.extend(self._merge_mcp_tools(managed_mcp_servers))
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
            smolagents, self.config.agent_type or DEFAULT_AGENT_TYPE
        )

        self._agent: MultiStepAgent = main_agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=tools,
            verbosity_level=-1,  # OFF
            managed_agents=managed_agents_instanced,
            **self.config.agent_args or {},
        )

        if self.config.instructions:
            self._agent.prompt_templates["system_prompt"] = self.config.instructions

    async def run_async(self, prompt: str) -> Any:
        """Run the Smolagents agent with the given prompt."""
        return self._agent.run(prompt)

    @property
    def tools(self) -> list[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._agent.tools
