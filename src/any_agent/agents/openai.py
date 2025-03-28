import os
from typing import Optional, Any

from loguru import logger

from any_agent.config import AgentFramework, AgentConfig
from any_agent.instructions import get_instructions
from any_agent.tools.wrappers import import_and_wrap_tools
from .any_agent import AnyAgent

try:
    from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

    agents_available = True
except ImportError:
    agents_available = None

OPENAI_MAX_TURNS = 30


class OpenAIAgent(AnyAgent):
    """OpenAI agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        self.managed_agents = managed_agents
        self.config = config
        self._load_agent()

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for an OpenAI agent."""
        if agent_config.api_key_var and agent_config.api_base:
            external_client = AsyncOpenAI(
                api_key=os.environ[agent_config.api_key_var],
                base_url=agent_config.api_base,
            )
            return OpenAIChatCompletionsModel(
                model=agent_config.model_id,
                openai_client=external_client,
            )
        return agent_config.model_id

    @logger.catch(reraise=True)
    def _load_agent(self) -> None:
        """Load the OpenAI agent with the given configuration."""
        if not agents_available:
            raise ImportError(
                "You need to `pip install openai-agents` to use this agent"
            )

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]
        tools, mcp_servers = import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.OPENAI
        )

        handoffs = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, managed_mcp_servers = import_and_wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.OPENAI
                )
                instance = Agent(
                    name=managed_agent.name,
                    instructions=get_instructions(managed_agent.instructions),
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    mcp_servers=[
                        managed_mcp_server.server
                        for managed_mcp_server in managed_mcp_servers
                    ],
                )
                if managed_agent.handoff:
                    handoffs.append(instance)
                else:
                    tools.append(
                        instance.as_tool(
                            tool_name=instance.name,
                            tool_description=managed_agent.description
                            or f"Use the agent: {managed_agent.name}",
                        )
                    )

        self.agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            handoffs=handoffs,
            tools=tools,
            mcp_servers=[mcp_server.server for mcp_server in mcp_servers],
        )

    @logger.catch(reraise=True)
    def run(self, prompt: str) -> Any:
        """Run the OpenAI agent with the given prompt."""
        if not agents_available:
            raise ImportError(
                "You need to `pip install openai-agents` to use this agent"
            )

        result = Runner.run_sync(self.agent, prompt, max_turns=OPENAI_MAX_TURNS)
        logger.info(result.final_output)
        return result
