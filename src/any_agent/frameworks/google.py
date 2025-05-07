from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

from .any_agent import AnyAgent

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types

    DEFAULT_MODEL_TYPE = LiteLlm
    adk_available = True
except ImportError:
    adk_available = False

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

    from any_agent.tracing.trace import AgentTrace


class GoogleAgent(AnyAgent):
    """Google ADK agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: Sequence[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
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

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]
        tools, _ = await self._load_tools(self.config.tools)

        sub_agents_instanced = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, _ = await self._load_tools(managed_agent.tools)

                agent_type = managed_agent.agent_type or LlmAgent

                managed_agent_args = managed_agent.agent_args or {}
                handoff = managed_agent_args.pop("handoff", None)
                instance = agent_type(
                    name=managed_agent.name,
                    instruction=managed_agent.instructions or "",
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    **managed_agent_args or {},
                )

                if handoff:
                    sub_agents_instanced.append(instance)
                else:
                    tools.append(AgentTool(instance))
        agent_type = self.config.agent_type or LlmAgent
        self._agent = agent_type(
            name=self.config.name,
            instruction=self.config.instructions or "",
            model=self._get_model(self.config),
            tools=tools,
            sub_agents=sub_agents_instanced,
            **self.config.agent_args or {},
            output_key="response",
        )

    async def run_async(  # type: ignore[no-untyped-def]
        self,
        prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> "AgentTrace":
        """Run the Google agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        runner = InMemoryRunner(self._agent)
        user_id = user_id or str(uuid4())
        session_id = session_id or str(uuid4())
        runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        events = runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            **kwargs,
        )

        async for event in events:
            logger.debug(event)
            if event.is_final_response():
                break

        session = runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        assert session, "Session should not be None"
        response = session.state.get("response", None)

        self._exporter.trace.final_output = response
        return self._exporter.trace
