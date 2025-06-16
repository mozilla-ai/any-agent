from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent

try:
    from smolagents import FinalAnswerTool, LiteLLMModel, ToolCallingAgent

    smolagents_available = True
except ImportError:
    smolagents_available = False

if TYPE_CHECKING:
    from smolagents import MultiStepAgent


DEFAULT_AGENT_TYPE = ToolCallingAgent
DEFAULT_MODEL_TYPE = LiteLLMModel


class SmolagentsAgent(AnyAgent):
    """Smolagents agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
    ):
        super().__init__(config)
        self._agent: MultiStepAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def _get_model(self, agent_config: AgentConfig) -> Any:
        """Get the model configuration for a smolagents agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        kwargs = {
            "model_id": agent_config.model_id,
            "api_key": agent_config.api_key,
            "api_base": agent_config.api_base,
            **model_args,
        }
        return model_type(**kwargs)

    async def _load_agent(self) -> None:
        """Load the Smolagents agent with the given configuration."""
        if not smolagents_available:
            msg = "You need to `pip install 'any-agent[smolagents]'` to use this agent"
            raise ImportError(msg)

        tools, _ = await self._load_tools(self.config.tools)

        main_agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE

        agent_args = self.config.agent_args or {}

        self._tools = tools
        self._agent = main_agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=tools,
            verbosity_level=-1,  # OFF
            **agent_args,
        )

        if self.config.instructions:
            self._agent.prompt_templates["system_prompt"] = self.config.instructions

        if self.config.output_type:
            output_type = self.config.output_type

            class CustomFinalAnswerTool(FinalAnswerTool):  # type: ignore[no-untyped-call]
                inputs = {  # noqa: RUF012
                    "answer": {
                        "type": "string",
                        "description": f"The final answer to the problem. The input must be a string that conforms to the{output_type.__name__} object.",
                    }
                }

                def forward(self, answer: str) -> Any:
                    output_type.model_validate_json(answer)
                    return answer

            self._agent.tools["final_answer"] = CustomFinalAnswerTool()  # type: ignore[no-untyped-call]

            self._agent.prompt_templates[
                "system_prompt"
            ] += f"""\n\nYour final answer must be a {self.config.output_type.__name__} object.
            This object must match the following schema:
            {self.config.output_type.model_json_schema()}
            """
        assert self._agent

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result = self._agent.run(prompt, **kwargs)
        if self.config.output_type:
            return self.config.output_type.model_validate_json(result)
        return str(result)
