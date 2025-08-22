import math
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

        tools = await self._load_tools(self.config.tools)

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
            mcp_servers=[],  # No longer needed with unified approach
            **kwargs_,
        )

    def _filter_mcp_tools(self, tools: list[Any], mcp_clients: list[Any]) -> list[Any]:
        """OpenAI framework doesn't expect the mcp tool to be included in `tools`."""
        # With the new MCPClient approach, MCP tools are already converted to regular callables
        # and included in the tools list, so we don't need to filter them out anymore.
        # The OpenAI framework can handle them as regular tools.
        return tools

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        if not kwargs.get("max_turns"):
            kwargs["max_turns"] = math.inf
        result = await Runner.run(self._agent, prompt, **kwargs)
        return result.final_output  # type: ignore[no-any-return]

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, we need to recreate it with the new output type
        # The OpenAI agents library requires output_type to be set during construction
        if self._agent:
            # Store current state
            current_tools = self._tools

            # Recreate the agent with the new output type
            kwargs_ = self.config.agent_args or {}
            if self.config.model_args:
                kwargs_["model_settings"] = ModelSettings(**self.config.model_args)
            if output_type:
                kwargs_["output_type"] = output_type

            self._agent = Agent(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self._get_model(self.config),
                tools=current_tools,
                mcp_servers=[],  # No longer needed with unified approach
                **kwargs_,
            )
