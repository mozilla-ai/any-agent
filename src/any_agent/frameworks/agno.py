from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    from agno.agent import Agent
    from agno.models.litellm import LiteLLM
    from agno.tools.toolkit import Toolkit

    DEFAULT_MODEL_TYPE = LiteLLM
    agno_available = True
except ImportError:
    agno_available = False


if TYPE_CHECKING:
    from agno.agent import RunResponse
    from agno.models.base import Model


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: Agent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.AGNO

    def _get_model(self, agent_config: AgentConfig) -> "Model":
        """Get the model configuration for an Agno agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE

        return model_type(
            id=agent_config.model_id,
            api_base=agent_config.api_base,
            api_key=agent_config.api_key,
            request_params=agent_config.model_args or {},  # type: ignore[arg-type]
        )

    @staticmethod
    def _unpack_tools(tools: list[Any]) -> list[Any]:
        unpacked: list[Any] = []
        for tool in tools:
            if isinstance(tool, Toolkit):
                unpacked.extend(f for f in tool.functions.values())
            else:
                unpacked.append(tool)
        return unpacked

    async def _load_agent(self) -> None:
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)

        tools, _ = await self._load_tools(self.config.tools)

        self._tools = self._unpack_tools(tools)

        agent_args = self.config.agent_args or {}
        if self.config.output_type:
            agent_args["response_model"] = self.config.output_type
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            tools=tools,
            **agent_args,
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result: RunResponse = await self._agent.arun(prompt, **kwargs)
        return result.content  # type: ignore[return-value]
