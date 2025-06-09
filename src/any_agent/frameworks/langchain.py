from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    from langchain_core.language_models import LanguageModelLike
    from langchain_litellm import ChatLiteLLM

    # Patch the _OPENAI_MODELS list to include additional models
    # This can be removed after https://github.com/Akshay-Dongare/langchain-litellm/issues/7
    from langchain_litellm.chat_models import litellm as langchain_litellm_module
    from langgraph.prebuilt import create_react_agent

    # Add your custom models to the existing list
    additional_models = ["gpt-4.1-mini", "gpt-4.1-nano"]

    # Extend the existing _OPENAI_MODELS list
    if hasattr(langchain_litellm_module, "_OPENAI_MODELS"):
        langchain_litellm_module._OPENAI_MODELS.extend(additional_models)

    DEFAULT_AGENT_TYPE = create_react_agent
    DEFAULT_MODEL_TYPE = ChatLiteLLM

    langchain_available = True
except ImportError:
    langchain_available = False

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.messages.base import BaseMessage
    from langgraph.graph.graph import CompiledGraph


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: CompiledGraph | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LANGCHAIN

    def _get_model(self, agent_config: AgentConfig) -> "LanguageModelLike":
        """Get the model configuration for a LangChain agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        return cast(
            "LanguageModelLike",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                model_kwargs=model_args,  # type: ignore[arg-type]
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
        if self.config.output_type:
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
        if self.config.output_type:
            structured_response = result.get("structured_response")
            if not structured_response:
                msg = "No structured output returned from the agent."
                raise ValueError(msg)
            return structured_response  # type: ignore[no-any-return]
        if not result.get("messages"):
            msg = "No messages returned from the agent."
            raise ValueError(msg)
        last_message: BaseMessage = result["messages"][-1]
        return str(last_message.content)
