import os
from typing import Any

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import wrap_tools

try:
    from agents import (
        Agent,
        AsyncOpenAI,
        ModelSettings,
        OpenAIChatCompletionsModel,
        Runner,
    )

    agents_available = True
except ImportError:
    agents_available = None

OPENAI_MAX_TURNS = 30


class OpenAIAgent(AnyAgent):
    """OpenAI agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: list[AgentConfig] | None = None
    ):
        if not agents_available:
            msg = "You need to `pip install 'any-agent[openai]'` to use this agent"
            raise ImportError(msg)
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None

    def _get_model(
        self,
        agent_config: AgentConfig,
        api_key_var: str | None = None,
        base_url: str | None = None,
    ):
        """Get the model configuration for an OpenAI agent."""
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
            msg = "You need to `pip install openai-agents` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]
        tools, mcp_servers = await wrap_tools(
            self.config.tools, agent_framework=AgentFramework.OPENAI
        )

        handoffs = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, managed_mcp_servers = await wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.OPENAI
                )
                kwargs = {}
                api_key_var = None
                base_url = None
                if managed_agent.model_args:
                    api_key_var = managed_agent.model_args.pop("api_key_var", None)
                    base_url = managed_agent.model_args.pop("base_url", None)
                    kwargs["model_settings"] = managed_agent.model_args
                instance = Agent(
                    name=managed_agent.name,
                    instructions=managed_agent.instructions,
                    model=self._get_model(managed_agent, api_key_var, base_url),
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
        api_key_var = None
        base_url = None
        if self.config.model_args:
            api_key_var = self.config.model_args.pop("api_key_var", None)
            base_url = self.config.model_args.pop("base_url", None)
            kwargs["model_settings"] = ModelSettings(**self.config.model_args)
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config, api_key_var, base_url),
            handoffs=handoffs,
            tools=tools,
            mcp_servers=[mcp_server.server for mcp_server in mcp_servers],
            **kwargs,
        )

    async def run_async(self, prompt: str) -> Any:
        """Run the OpenAI agent with the given prompt asynchronously."""
        return await Runner.run(self._agent, prompt, max_turns=OPENAI_MAX_TURNS)

    @property
    def tools(self) -> list[str]:
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
                    msg = f"No tools found in MCP {server_name}"
                    raise ValueError(msg)
        else:
            logger.warning("Agent not loaded or does not have tools.")
            tools = []
        return tools
