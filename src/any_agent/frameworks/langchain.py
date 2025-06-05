from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_agent.config import (
    AgentConfig,
    AgentFramework,
    DefaultAgentOutput,
    TracingConfig,
)

from .any_agent import AnyAgent

try:
    from langchain_core.language_models import LanguageModelLike
    from langchain_litellm import ChatLiteLLM
    from langgraph.prebuilt import create_react_agent

    DEFAULT_AGENT_TYPE = create_react_agent
    DEFAULT_MODEL_TYPE = ChatLiteLLM

    langchain_available = True
except ImportError:
    langchain_available = False

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langgraph.graph.graph import CompiledGraph


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, tracing)
        self._agent: CompiledGraph | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LANGCHAIN

    def _get_model(self, agent_config: AgentConfig) -> "LanguageModelLike":
        """Get the model configuration for a LangChain agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE

        return cast(
            "LanguageModelLike",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                model_kwargs=agent_config.model_args or {},  # type: ignore[arg-type]
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""
        if not langchain_available:
            msg = "You need to `pip install 'any-agent[langchain]'` to use this agent"
            raise ImportError(msg)

        imported_tools, _ = await self._load_tools(self.config.tools)

        self._tools = imported_tools
        agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
        agent_args = self.config.agent_args or {}
        agent_args["response_format"] = self.config.output_type
        self._agent = agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=imported_tools,
            prompt=self.config.instructions,
            **agent_args,
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        inputs = {"messages": [("user", prompt)]}
        result = await self._agent.ainvoke(inputs, **kwargs)
        structured_response = result.get("structured_response")
        if not structured_response:
            msg = "No structured output returned from the agent."
            raise ValueError(msg)
        if isinstance(structured_response, DefaultAgentOutput):
            return structured_response.answer
        return structured_response  # type: ignore[no-any-return]
