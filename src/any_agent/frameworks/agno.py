from typing import Any, List, Optional
from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools.wrappers import import_and_wrap_tools

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM

    agno_available = True
except ImportError:
    agno_available = None


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not agno_available:
            raise ImportError(
                "You need to `pip install 'any-agent[agno]'` to use this agent"
            )
        if managed_agents:
            raise NotImplementedError(
                "Managed agents are not yet supported in Agno agent."
            )
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
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]
        tools, mcp_servers = await import_and_wrap_tools(
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
        result = await self._agent.arun(prompt)
        return result

    @property
    def tools(self) -> List[str]:
        if hasattr(self, "_agent"):
            tools = self._agent.tools
        else:
            logger.warning("Agent not loaded or does not have tools.")
            tools = []
        return tools
