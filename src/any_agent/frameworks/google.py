from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

try:
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types

    adk_available = True
except ImportError:
    adk_available = False


class GoogleAgent(AnyAgent):
    """Google ADK agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: Sequence[AgentConfig] | None = None,
    ):
        super().__init__(config, managed_agents)
        self._agent: Agent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.GOOGLE

    def _get_model(self, agent_config: AgentConfig) -> LiteLlm:
        """Get the model configuration for a Google agent."""
        return LiteLlm(
            model=agent_config.model_id,
            api_key=agent_config.api_key,
            api_base=agent_config.api_base,
            **agent_config.model_args or {},
        )

    async def load_agent(self) -> None:
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

                instance = Agent(
                    name=managed_agent.name,
                    instruction=managed_agent.instructions or "",
                    model=self._get_model(managed_agent),
                    tools=managed_tools,
                    **managed_agent.agent_args or {},
                )

                if managed_agent.handoff:
                    sub_agents_instanced.append(instance)
                else:
                    tools.append(AgentTool(instance))

        self._agent = Agent(
            name=self.config.name,
            instruction=self.config.instructions or "",
            model=self._get_model(self.config),
            tools=tools,
            sub_agents=sub_agents_instanced,  # type: ignore[arg-type]
            **self.config.agent_args or {},
            output_key="response",
        )

    async def run_async(
        self,
        prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Run the Google agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)

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
        return session.state.get("response", None)
