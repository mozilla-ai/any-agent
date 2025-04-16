from typing import Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import wrap_tools

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM

    agno_available = True
except ImportError:
    agno_available = None


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: list[AgentConfig] | None = None
    ):
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)
        if managed_agents:
            msg = "Managed agents are not yet supported in Agno agent."
            raise NotImplementedError(msg)
        self.managed_agents = managed_agents  # Future proofing
        self.config = config
        self._agent = None
        self._mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for an Agno agent."""
        return LiteLLM(
            id=agent_config.model_id,
            **agent_config.model_args or {},
        )

    async def _load_agent(self) -> None:
        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]
        tools, mcp_servers = await wrap_tools(
            self.config.tools, agent_framework=AgentFramework.AGNO
        )
        # Add to agent so that it doesn't get garbage collected
        self._mcp_servers = mcp_servers
        mcp_tools = [mcp_server.tools for mcp_server in mcp_servers]
        tools.extend(mcp_tools)

        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions or "",
            model=self._get_model(self.config),
            tools=tools,
            **self.config.agent_args or {},
        )

    async def run_async(self, prompt: str) -> Any:
        return await self._agent.arun(prompt)

    @property
    def tools(self) -> list[str]:
        if hasattr(self, "_agent"):
            tools = self._agent.tools
        else:
            logger.warning("Agent not loaded or does not have tools.")
            tools = []
        return tools
