import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.frameworks.any_agent import AgentResult, AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph


if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.messages.base import BaseMessage
try:
    from langchain_core.language_models import LanguageModelLike
    from langgraph.prebuilt import create_react_agent
    from langgraph_swarm import create_handoff_tool, create_swarm

    langchain_available = True
except ImportError:
    langchain_available = False


DEFAULT_MODEL_CLASS = "langchain_litellm.ChatLiteLLM"


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

    def _get_model(self, agent_config: AgentConfig) -> str | LanguageModelLike:
        """Get the model configuration for a LangChain agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(importlib.import_module(module), class_name)

        return cast(
            "str | LanguageModelLike",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                **agent_config.model_args or {},
            ),
        )

    async def load_agent(self) -> None:
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

                managed_agent = create_react_agent(  # type: ignore[assignment]
                    name=name,
                    model=self._get_model(managed_agent),
                    tools=[
                        *managed_tools,
                        create_handoff_tool(agent_name=self.config.name),
                    ],
                    prompt=managed_agent.instructions,
                    **managed_agent.agent_args or {},
                )
                swarm.append(managed_agent)

            imported_tools = [
                create_handoff_tool(agent_name=managed_name)
                for managed_name in managed_names
            ]

            main_agent = create_react_agent(
                name=self.config.name,
                model=self._get_model(self.config),
                tools=imported_tools,
                prompt=self.config.instructions,
                **self.config.agent_args or {},
            )
            swarm.append(main_agent)  # type: ignore[arg-type]
            workflow = create_swarm(swarm, default_active_agent=self.config.name)  # type: ignore[arg-type]
            self._agent = workflow.compile()
        else:
            self._agent = create_react_agent(
                name=self.config.name,
                model=self._get_model(self.config),
                tools=imported_tools,
                prompt=self.config.instructions,
                **self.config.agent_args or {},
            )
        # Langgraph doesn't let you easily access what tools are loaded from the CompiledGraph,
        # so we'll store a list of them in this class
        self._tools = imported_tools

    async def run_async(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Run the LangChain agent with the given prompt."""
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._create_tracer()
        inputs = {"messages": [("user", prompt)]}
        result = await self._agent.ainvoke(inputs, **kwargs)
        if not result.get("messages"):
            msg = "No messages returned from the agent."
            raise ValueError(msg)
        last_message: BaseMessage = result["messages"][-1]
        return AgentResult(
            final_output=last_message.content,
            raw_responses=result["messages"],
            trace=self._get_trace(),
        )
