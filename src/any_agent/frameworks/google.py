from typing import Optional, Any, List
from uuid import uuid4

from any_agent.config import AgentFramework, AgentConfig
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.instructions import get_instructions
from any_agent.logging import logger
from any_agent.tools.wrappers import import_and_wrap_tools

try:
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types

    adk_available = True
except ImportError:
    adk_available = None


class GoogleAgent(AnyAgent):
    """Google agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not adk_available:
            raise ImportError(
                "You need to `pip install 'any-agent[google]'` to use this agent"
            )
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        self._mcp_servers = None
        self._managed_mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a Google agent."""
        return LiteLlm(model=agent_config.model_id, **agent_config.model_args or {})

    async def _load_agent(self) -> None:
        """Load the Google agent with the given configuration."""
        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]
        tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.GOOGLE
        )
        # Add to agent so that it doesn't get garbage collected
        self._mcp_servers = mcp_servers
        mcp_tools = [tool for mcp_server in mcp_servers for tool in mcp_server.tools]
        tools.extend(mcp_tools)

        sub_agents_instanced = []
        if self.managed_agents:
            for managed_agent in self.managed_agents:
                managed_tools, managed_mcp_servers = await import_and_wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.GOOGLE
                )
                # Add to agent so that it doesn't get garbage collected
                self._managed_mcp_servers = managed_mcp_servers
                managed_mcp_tools = [
                    tool
                    for mcp_server in managed_mcp_servers
                    for tool in mcp_server.tools
                ]
                managed_tools.extend(managed_mcp_tools)
                instance = Agent(
                    name=managed_agent.name,
                    instruction=get_instructions(managed_agent.instructions) or "",
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
            sub_agents=sub_agents_instanced,
            **self.config.agent_args or {},
            output_key="response",
        )

    async def run_async(
        self, prompt: str, user_id: str | None = None, session_id: str | None = None
    ) -> Any:
        """Run the Google agent with the given prompt."""
        runner = InMemoryRunner(self._agent)
        user_id = user_id or str(uuid4())
        session_id = session_id or str(uuid4())
        runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
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
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        return session.state.get("response", None)

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        if hasattr(self, "_agent"):
            tools = [tool.name for tool in self._agent.tools]
        else:
            logger.warning("Agent not loaded or does not have tools.")
            tools = []
        return tools
