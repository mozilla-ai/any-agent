from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

from .any_agent import AnyAgent

try:
    from langchain_core.language_models import LanguageModelLike
    from langchain_litellm import ChatLiteLLM
    from langgraph.prebuilt import create_react_agent
    from langgraph_swarm import create_handoff_tool, create_swarm

    DEFAULT_AGENT_TYPE = create_react_agent
    DEFAULT_MODEL_TYPE = ChatLiteLLM

    langchain_available = True
except ImportError:
    langchain_available = False

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.messages.base import BaseMessage
    from langgraph.graph.graph import CompiledGraph

    from any_agent.tracing.trace import AgentTrace


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: Sequence[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
        self._agent: CompiledGraph | None = None
        self._tools: Sequence[Any] = []

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LANGCHAIN

    def _get_model(self, agent_config: AgentConfig) -> "LanguageModelLike":
        """Get the model configuration for a LangChain agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE

        return cast(
            "LanguageModelLike",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                **agent_config.model_args or {},
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""
        if not langchain_available:
            msg = "You need to `pip install 'any-agent[langchain]'` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        imported_tools, _ = await self._load_tools(self.config.tools)

        if self.managed_agents:
            swarm = []
            managed_names = []
            for n, managed_agent in enumerate(self.managed_agents):
                managed_tools, _ = await self._load_tools(managed_agent.tools)
                name = managed_agent.name
                if not name or name == "any_agent":
                    logger.warning(
                        "Overriding name for managed_agent. Can't use the default.",
                    )
                    name = f"managed_agent_{n}"
                managed_names.append(name)

                agent_type = managed_agent.agent_type or DEFAULT_AGENT_TYPE
                instance = agent_type(
                    name=name,
                    model=self._get_model(managed_agent),
                    tools=[
                        *managed_tools,
                        create_handoff_tool(agent_name=self.config.name),
                    ],
                    prompt=managed_agent.instructions,
                    **managed_agent.agent_args or {},
                )
                swarm.append(instance)

            imported_tools = [
                create_handoff_tool(agent_name=managed_name)
                for managed_name in managed_names
            ]
            agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
            main_agent = agent_type(
                name=self.config.name,
                model=self._get_model(self.config),
                tools=imported_tools,
                prompt=self.config.instructions,
                **self.config.agent_args or {},
            )
            swarm.append(main_agent)
            workflow = create_swarm(swarm, default_active_agent=self.config.name)
            self._agent = workflow.compile()
        else:
            agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
            self._agent = agent_type(
                name=self.config.name,
                model=self._get_model(self.config),
                tools=imported_tools,
                prompt=self.config.instructions,
                **self.config.agent_args or {},
            )
        # Langgraph doesn't let you easily access what tools are loaded from the CompiledGraph,
        # so we'll store a list of them in this class
        self._tools = imported_tools

    async def run_async(self, prompt: str, **kwargs: Any) -> "AgentTrace":
        """Run the LangChain agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        inputs = {"messages": [("user", prompt)]}
        result = await self._agent.ainvoke(inputs, **kwargs)
        if not result.get("messages"):
            msg = "No messages returned from the agent."
            raise ValueError(msg)
        last_message: BaseMessage = result["messages"][-1]
        self._exporter.trace.final_output = str(last_message.content)
        return self._exporter.trace
