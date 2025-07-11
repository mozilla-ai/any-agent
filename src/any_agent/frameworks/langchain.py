from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm.utils import supports_response_schema
from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    from langchain_core.language_models import LanguageModelLike
    from langgraph.prebuilt import create_react_agent

    from any_agent.vendor.langchain_litellm import ChatLiteLLM, _convert_message_to_dict

    DEFAULT_AGENT_TYPE = create_react_agent
    DEFAULT_MODEL_TYPE = ChatLiteLLM

    langchain_available = True
except ImportError:
    langchain_available = False

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langgraph.graph.state import CompiledStateGraph


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: CompiledStateGraph[Any] | None = None

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

        if not result.get("messages"):
            msg = "No messages returned from the agent."
            raise ValueError(msg)

        # Post-process for structured output if needed
        # This emulates the langgraph behavior for structured outputs,
        # but since it happens outside of langgraph we can control the model call,
        # because some providers like mistral don't allow for the model to be called without the most recent message being a user message.
        if self.config.output_type:
            # Add a follow-up message to request structured output
            structured_output_message = {
                "role": "user",
                "content": f"Please conform your output to the following schema: {self.config.output_type.model_json_schema()}.",
            }
            completion_params: dict[str, Any] = {}
            if self.config.model_args:
                # only include the temperature and frequency_penalty, not anything related to tools
                completion_params["temperature"] = self.config.model_args.get(
                    "temperature"
                )
                completion_params["frequency_penalty"] = self.config.model_args.get(
                    "frequency_penalty"
                )

            completion_params["model"] = self.config.model_id
            previous_messages = [
                _convert_message_to_dict(m) for m in result["messages"]
            ]
            completion_params["messages"] = [
                *previous_messages,
                structured_output_message,
            ]

            # Use response schema if supported by the model
            if supports_response_schema(model=self.config.model_id):
                completion_params["response_format"] = self.config.output_type

            response = await self.call_model(**completion_params)
            return self.config.output_type.model_validate_json(
                response.choices[0].message["content"]
            )
        return str(result["messages"][-1].content)

    async def call_model(self, **kwargs: Any) -> Any:
        return await litellm.acompletion(**kwargs)
