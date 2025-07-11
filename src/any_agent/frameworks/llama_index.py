from typing import TYPE_CHECKING, Any, cast

from litellm import acompletion
from litellm.utils import supports_response_schema
from pydantic import BaseModel, ValidationError

from any_agent import AgentConfig, AgentFramework
from any_agent.logging import logger
from any_agent.tools.final_output import prepare_final_output

from .any_agent import AnyAgent

try:
    from llama_index.core.agent.workflow import (
        BaseWorkflowAgent,
        FunctionAgent,
        ReActAgent,
    )
    from llama_index.llms.litellm import LiteLLM

    DEFAULT_AGENT_TYPE = FunctionAgent
    DEFAULT_MODEL_TYPE = LiteLLM
    llama_index_available = True
except ImportError:
    llama_index_available = False


if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.llms import LLM


class LlamaIndexAgent(AnyAgent):
    """LLamaIndex agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: BaseWorkflowAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LLAMA_INDEX

    def _get_model(self, agent_config: AgentConfig) -> "LLM":
        """Get the model configuration for a llama_index agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        additional_kwargs = agent_config.model_args or {}
        additional_kwargs["stream_options"] = {
            "include_usage": True
        }  # Needed so that we get usage stats
        return cast(
            "LLM",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                additional_kwargs=additional_kwargs,  # type: ignore[arg-type]
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LLamaIndex agent with the given configuration."""
        if not llama_index_available:
            msg = "You need to `pip install 'any-agent[llama_index]'` to use this agent"
            raise ImportError(msg)

        instructions = self.config.instructions
        tools_to_use = list(self.config.tools)
        if self.config.output_type:
            instructions, final_output_function = prepare_final_output(
                self.config.output_type, instructions
            )
            tools_to_use.append(final_output_function)
        imported_tools, _ = await self._load_tools(tools_to_use)
        agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
        # if agent type is FunctionAgent but there are no tools, throw an error
        if agent_type == FunctionAgent and not imported_tools:
            logger.warning(
                "FunctionAgent requires tools and none were provided. Using ReActAgent instead."
            )
            agent_type = ReActAgent
        self._tools = imported_tools

        self._agent = agent_type(
            name=self.config.name,
            tools=imported_tools,
            description=self.config.description or "The main agent",
            llm=self._get_model(self.config),
            system_prompt=instructions,
            **self.config.agent_args or {},
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result: AgentOutput = await self._agent.run(prompt, **kwargs)
        # assert that it's a TextBlock
        if not result.response.blocks or not hasattr(result.response.blocks[0], "text"):
            msg = f"Agent did not return a valid response: {result.response}"
            raise ValueError(msg)
        if self.config.output_type:
            # First try to validate the output directly
            try:
                return self.config.output_type.model_validate_json(
                    result.response.blocks[0].text
                )
            except ValidationError:
                # If validation fails, send it through litellm to enforce structured output
                completion_params = self.config.model_args or {}
                completion_params["model"] = self.config.model_id
                model_output_message = {
                    "role": "assistant",
                    "content": result.response.blocks[0].text,
                }
                structured_output_message = {
                    "role": "user",
                    "content": f"Please conform your output to the following schema: {self.config.output_type.model_json_schema()}.",
                }
                completion_params["messages"] = [
                    model_output_message,
                    structured_output_message,
                ]
                if supports_response_schema(model=self.config.model_id):
                    completion_params["response_format"] = self.config.output_type
                response = await acompletion(**completion_params)
                return self.config.output_type.model_validate_json(
                    response.choices[0].message["content"]
                )
        return result.response.blocks[0].text
