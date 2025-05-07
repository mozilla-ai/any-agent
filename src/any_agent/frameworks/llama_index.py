from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from any_agent import AgentConfig, AgentFramework
from any_agent.config import TracingConfig
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage

from .any_agent import AnyAgent

try:
    from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
    from llama_index.llms.litellm import LiteLLM

    DEFAULT_AGENT_TYPE = ReActAgent
    DEFAULT_MODEL_TYPE = LiteLLM
    llama_index_available = True
except ImportError:
    llama_index_available = False


if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.llms import LLM

    from any_agent.tracing.trace import AgentTrace


class LlamaIndexAgent(AnyAgent):
    """LLamaIndex agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: Sequence[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ):
        super().__init__(config, managed_agents, tracing)
        self._agent: AgentWorkflow | ReActAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LLAMA_INDEX

    def _get_model(self, agent_config: AgentConfig) -> "LLM":
        """Get the model configuration for a llama_index agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        return cast(
            "LLM",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                **agent_config.model_args or {},
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LLamaIndex agent with the given configuration."""
        if not llama_index_available:
            msg = "You need to `pip install 'any-agent[llama_index]'` to use this agent"
            raise ImportError(msg)

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        if self.managed_agents:
            agents = []
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
                managed_instance = agent_type(
                    name=name,
                    description=managed_agent.description or "A managed agent",
                    system_prompt=managed_agent.instructions,
                    tools=managed_tools,
                    llm=self._get_model(managed_agent),
                    can_handoff_to=[self.config.name],
                    **managed_agent.agent_args or {},
                )
                agents.append(managed_instance)

            main_tools, _ = await self._load_tools(self.config.tools)
            agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
            main_agent = agent_type(
                name=self.config.name,
                description=self.config.description or "The main agent",
                tools=main_tools,
                llm=self._get_model(self.config),
                system_prompt=self.config.instructions,
                can_handoff_to=managed_names,
                **self.config.agent_args or {},
            )
            agents.append(main_agent)

            self._agent = AgentWorkflow(agents=agents, root_agent=main_agent.name)

        else:
            imported_tools, _ = await self._load_tools(self.config.tools)
            agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
            self._agent = agent_type(
                name=self.config.name,
                tools=imported_tools,
                description=self.config.description or "The main agent",
                llm=self._get_model(self.config),
                system_prompt=self.config.instructions,
                **self.config.agent_args or {},
            )

    async def run_async(self, prompt: str, **kwargs: Any) -> "AgentTrace":
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        self._setup_tracing()
        result: AgentOutput = await self._agent.run(prompt, **kwargs)
        # assert that it's a TextBlock
        if not result.response.blocks or not hasattr(result.response.blocks[0], "text"):
            msg = f"Agent did not return a valid response: {result.response}"
            raise ValueError(msg)

        self._exporter.trace.final_output = result.response.blocks[0].text
        return self._exporter.trace
