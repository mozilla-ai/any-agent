from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from any_agent.config import AgentConfig, AgentFramework, TracingConfig

from .any_agent import AnyAgent

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions.session import Session
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
        return model_type(
            model=agent_config.model_id,
            api_key=agent_config.api_key,
            api_base=agent_config.api_base,
            **agent_config.model_args or {},
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
            # The design of structured output in the ADK is a little different from other frameworks.
            # There's a useful discussion here https://github.com/google/adk-python/discussions/322
            # The main difference is that in the ADK, if you want an agent to return a specific schema, it is not
            # allowed to use any tools.
            # https://google.github.io/adk-docs/agents/llm-agents/#structuring-data-input_schema-output_schema-output_key
            # In order to work around this, we'll append instructions about the output schema to the instructions,
            # and we'll also use the validate_final_output tool to validate the output, with a few build in retries.
            instructions += f"""\n\nYou must return a {self.config.output_type.__name__} object.
            This object must match the following schema:
            {self.config.output_type.model_json_schema()}
            """

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
        return await self._validate_final_output(runner, session)

    async def _validate_final_output(
        self,
        runner: InMemoryRunner,
        session: Session,
        attempt: int = 0,
    ) -> str | BaseModel:
        response = str(session.state.get("response"))
        if self.config.output_type:
            try:
                return self.config.output_type.model_validate_json(response)
            except ValidationError as e:
                if attempt >= DEFAULT_MAX_SCHEMA_VALIDATION_ATTEMPTS:
                    msg = f"Failed to generate appropriate final answer schema after {DEFAULT_MAX_SCHEMA_VALIDATION_ATTEMPTS} attempts"
                    raise ValueError(msg) from e
                output_type_schema = self.config.output_type.model_json_schema()
                fix_prompt = f"""
                The response is invalid. Please fix it. The error is: {e}
                The output type schema is: {output_type_schema}
                The response was: {response}.
                Do not return anything else other than the fixed JSON object.
                """
                async for _ in runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=fix_prompt)]
                    ),
                ):
                    pass
                new_session = await runner.session_service.get_session(
                    app_name=runner.app_name,
                    user_id=session.user_id,
                    session_id=session.id,
                )
                assert new_session, "Session should not be None"
                return await self._validate_final_output(
                    runner, new_session, attempt + 1
                )
        return response
