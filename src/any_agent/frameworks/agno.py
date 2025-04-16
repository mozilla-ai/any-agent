from collections.abc import Sequence
from typing import Any

from any_agent.config import AgentConfig, AgentFramework, Tool
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools.mcp import MCPServerBase
from any_agent.tools.wrappers import import_and_wrap_tools

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM

    agno_available = True
except ImportError:
    agno_available = False


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
    ):
        if not agno_available:
            raise ImportError(
                "You need to `pip install 'any-agent[agno]'` to use this agent",
            )
        if managed_agents:
            raise NotImplementedError(
                "Managed agents are not yet supported in Agno agent.",
            )
        self.managed_agents = managed_agents  # Future proofing
        self.config = config
        self._agent: Agent | None = None
        self._mcp_servers: Sequence[MCPServerBase] | None = None

    def _get_model(self, agent_config: AgentConfig) -> LiteLLM:
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
            self.config.tools,
            agent_framework=AgentFramework.AGNO,
        )
        # Add to agent so that it doesn't get garbage collected
        self._mcp_servers = mcp_servers
        for mcp_server in mcp_servers:
            tools.extend(mcp_server.tools)

        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions or "",
            model=self._get_model(self.config),
            tools=tools,
            **self.config.agent_args or {},
        )

    async def run_async(self, prompt: str) -> Any:
        result = await self._agent.arun(prompt)  # type: ignore[union-attr]
        return result

    @property
    def tools(self) -> list[Tool]:
        if not self._agent:
            logger.warning("Agent not loaded or does not have tools.")
            return []

        return self._agent.tools  # type: ignore[no-any-return]
