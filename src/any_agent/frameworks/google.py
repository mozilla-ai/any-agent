from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from any_agent.config import AgentConfig, AgentFramework
from any_agent.tools.final_output import prepare_final_output

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


class GoogleAgent(AnyAgent):
    """Google ADK agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: LlmAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.GOOGLE

    def _get_model(self, agent_config: AgentConfig) -> "BaseLlm":
        """Get the model configuration for a Google agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        if self.config.output_type:
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
        if self.config.output_type:
            instructions, final_output_tool = prepare_final_output(
                self.config.output_type, instructions
            )
            tools.append(final_output_tool)

        self._agent = agent_type(
            name=self.config.name,
            instruction=instructions,
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

        if self.config.output_type:
            final_output = None
            final_output_attempts = 0
            # We allow for two retries: one to make it a proper json string, and one to make it a valid pydantic model
            max_output_attepts = 3

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                **kwargs,
            ):
                if not event.content or not event.content.parts:
                    continue

                # Check for final_output function responses
                for part in event.content.parts:
                    if (
                        part.function_response
                        and part.function_response.name == "final_output"
                        and part.function_response.response
                    ):
                        final_output_attempts += 1
                        if part.function_response.response.get("success"):
                            final_output = part.function_response.response.get("result")
                            break
                        if final_output_attempts >= max_output_attepts:
                            msg = f"Final output failed after {final_output_attempts} attempts"
                            raise ValueError(msg)

                if final_output or final_output_attempts >= max_output_attepts:
                    break

            if not final_output:
                msg = "No final response found"
                raise ValueError(msg)
            return self.config.output_type.model_validate_json(final_output)

        async for _ in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            **kwargs,
        ):
            pass
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        assert session, "Session should not be None"
        return str(session.state.get("response"))
