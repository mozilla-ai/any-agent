import os
from typing import Optional, Any, List

from any_agent.config import AgentFramework, AgentConfig
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.instructions import get_instructions
from any_agent.logging import logger
from any_agent.tools.wrappers import import_and_wrap_tools

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
        if not agents_available:
            raise ImportError(
                "You need to `pip install 'any-agent[openai]'` to use this agent"
            )
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for an OpenAI agent."""
        model_args = agent_config.model_args or {}
        api_key_var = model_args.pop("api_key_var", None)
        base_url = model_args.pop("base_url", None)
        if api_key_var and base_url:
            external_client = AsyncOpenAI(
                api_key=os.environ[api_key_var],
                base_url=base_url,
            )
            return OpenAIChatCompletionsModel(
                model=agent_config.model_id,
                openai_client=external_client,
            )
        return agent_config.model_id

    async def _load_agent(self) -> None:
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
        tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.OPENAI
        )

        handoffs = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, managed_mcp_servers = await import_and_wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.OPENAI
                )
                kwargs = {}
                if managed_agent.model_args:
                    kwargs["model_settings"] = managed_agent.model_args
                instance = Agent(
                    name=managed_agent.name,
                    instructions=get_instructions(managed_agent.instructions),
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    mcp_servers=[
                        managed_mcp_server.server
                        for managed_mcp_server in managed_mcp_servers
                    ],
                    **kwargs,
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

        kwargs = self.config.agent_args or {}
        if self.config.model_args:
            kwargs["model_settings"] = self.config.model_args
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            handoffs=handoffs,
            tools=tools,
            mcp_servers=[mcp_server.server for mcp_server in mcp_servers],
            **kwargs,
        )

    async def run_async(self, prompt: str) -> Any:
        """Run the OpenAI agent with the given prompt asynchronously."""
        result = await Runner.run(self._agent, prompt, max_turns=OPENAI_MAX_TURNS)
        return result

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        if hasattr(self, "_agent"):
            # Extract tool names from the agent's tools
            tools = [tool.name for tool in self._agent.tools]
            # Add MCP tools to the list
            for mcp_server in self._agent.mcp_servers:
                tools_in_mcp = mcp_server._tools_list
                server_name = mcp_server.name.replace(" ", "_")
                if tools_in_mcp:
                    tools.extend(
                        [f"{server_name}_{tool.name}" for tool in tools_in_mcp]
                    )
                else:
                    raise ValueError(f"No tools found in MCP {server_name}")
        else:
            logger.warning("Agent not loaded or does not have tools.")
            tools = []
        return tools
