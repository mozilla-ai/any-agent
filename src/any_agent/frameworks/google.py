from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from any_agent.config import (
    AgentConfig,
    AgentFramework,
    DefaultAgentOutput,
    TracingConfig,
)

from .any_agent import AnyAgent

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    DEFAULT_MODEL_TYPE = LiteLlm
    adk_available = True
except ImportError:
    adk_available = False

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

DEFAULT_MAX_SCHEMA_VALIDATION_ATTEMPTS = 3


class GoogleAgent(AnyAgent):
    """Google ADK agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, tracing)
        self._agent: LlmAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.GOOGLE

    def _get_model(self, agent_config: AgentConfig) -> "BaseLlm":
        """Get the model configuration for a Google agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        model_args["tool_choice"] = "required"
        return model_type(
            model=agent_config.model_id,
            api_key=agent_config.api_key,
            api_base=agent_config.api_base,
            **model_args,
        )

    async def _load_agent(self) -> None:
        """Load the Google agent with the given configuration."""
        if not adk_available:
            msg = "You need to `pip install 'any-agent[google]'` to use this agent"
            raise ImportError(msg)

        tools, _ = await self._load_tools(self.config.tools)

        agent_type = self.config.agent_type or LlmAgent

        self._tools = tools

        instructions = self.config.instructions or ""
        instructions += (
            "You must call the final_output tool when finished. "
            "The argument passed to the final_output tool must be a JSON string that matches the following schema:\n"
            f"{self.config.output_type.model_json_schema()}"
        )

        def final_output(final_output: str) -> dict:  # type: ignore[type-arg]
            try:
                self.config.output_type.model_validate_json(final_output)
            except ValidationError as e:
                return {
                    "success": False,
                    "result": f"Please fix this validation error: {e}. The format must conform to {self.config.output_type.model_json_schema()}",
                }
            else:
                return {"success": True, "result": final_output}

        final_output.__doc__ = f"""You must call this tool in order to return the final answer.

            Args:
                final_output: The final output that can be loaded as a Pydantic model. This must be a JSON compatible string that matches the following schema:
                    {self.config.output_type.model_json_schema()}

            Returns:
                A dictionary with the following keys:
                    - success: True if the output is valid, False otherwise.
                    - result: The final output if success is True, otherwise an error message.

            """
        tools.append(final_output)

        self._agent = agent_type(
            name=self.config.name,
            instruction=instructions,
            global_instruction=instructions,
            model=self._get_model(self.config),
            tools=tools,
            **self.config.agent_args or {},
            output_key="response",
        )

    async def _run_async(  # type: ignore[no-untyped-def]
        self,
        prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        runner = InMemoryRunner(self._agent)
        user_id = user_id or str(uuid4())
        session_id = session_id or str(uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        final_output = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            **kwargs,
        ):
            if not event.content or not event.content.parts:
                continue
            if any(
                part.function_response
                and part.function_response.name == "final_output"
                and part.function_response.response
                and part.function_response.response.get("success")
                for part in event.content.parts
            ):
                # Extract the final output from the successful function response
                for part in event.content.parts:
                    if (
                        part.function_response
                        and part.function_response.name == "final_output"
                        and part.function_response.response
                        and part.function_response.response.get("success")
                    ):
                        final_output = part.function_response.response.get("result")
                        break
                break

        if not final_output:
            msg = "No final response found"
            raise ValueError(msg)
        result = self.config.output_type.model_validate_json(final_output)
        if isinstance(result, DefaultAgentOutput):
            return result.answer
        return result
