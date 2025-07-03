from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    from agents import (
        Agent,
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


class OpenAIAgent(AnyAgent):
    """OpenAI agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
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

        tools, mcp_servers = await self._load_tools(self.config.tools)
        tools = self._filter_mcp_tools(tools, mcp_servers)

        kwargs_ = self.config.agent_args or {}
        if self.config.model_args:
            kwargs_["model_settings"] = ModelSettings(**self.config.model_args)
        if self.config.output_type:
            kwargs_["output_type"] = self.config.output_type

        self._tools = tools
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            tools=tools,
            mcp_servers=[mcp_server.server for mcp_server in mcp_servers],
            **kwargs_,
        )

    def _filter_mcp_tools(self, tools: list[Any], mcp_servers: list[Any]) -> list[Any]:
        """OpenAI framework doesn't expect the mcp tool to be included in `tools`."""
        non_mcp_tools = []
        for tool in tools:
            if any(tool in mcp_server.tools for mcp_server in mcp_servers):
                continue
            non_mcp_tools.append(tool)
        return non_mcp_tools

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result = await Runner.run(self._agent, prompt, **kwargs)
        return result.final_output  # type: ignore[no-any-return]
