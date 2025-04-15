import importlib
from typing import Optional, List

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.logging import logger
from any_agent.tools.wrappers import import_and_wrap_tools

try:
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.agent.workflow import AgentWorkflow

    llama_index_available = True
except ImportError:
    llama_index_available = False


DEFAULT_MODEL_CLASS = "litellm.LiteLLM"


class LlamaIndexAgent(AnyAgent):
    """LLamaIndex agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not llama_index_available:
            raise ImportError(
                "You need to `pip install 'any-agent[llama_index]'` to use this agent"
            )
        self.managed_agents: Optional[list[AgentConfig]] = managed_agents
        self.config: AgentConfig = config
        self._agent = None
        self._mcp_servers = []
        self.framework = AgentFramework.LLAMAINDEX

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a llama_index agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.rsplit(".")
        model_type = getattr(
            importlib.import_module(f"llama_index.llms.{module}"), class_name
        )
        return model_type(model=agent_config.model_id, **agent_config.model_args or {})

    async def _load_tools(self, tools):
        imported_tools, mcp_servers = await import_and_wrap_tools(tools, self.framework)

        # Add to agent so that it doesn't get garbage collected
        self._mcp_servers.extend(mcp_servers)

        for mcp_server in mcp_servers:
            imported_tools.extend(mcp_server.tools)

        return imported_tools

    async def _load_agent(self) -> None:
        """Load the LLamaIndex agent with the given configuration."""

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]

        if self.managed_agents:
            agents = []
            managed_names = []
            for n, managed_agent in enumerate(self.managed_agents):
                managed_tools = await self._load_tools(managed_agent.tools)
                name = managed_agent.name
                if not name or name == "any_agent":
                    logger.warning(
                        "Overriding name for managed_agent. Can't use the default."
                    )
                    name = f"managed_agent_{n}"
                managed_names.append(name)
                managed_instance = ReActAgent(
                    name=name,
                    description=managed_agent.description,
                    system_prompt=managed_agent.instructions,
                    tools=managed_tools,
                    llm=self._get_model(managed_agent),
                    can_handoff_to=[self.config.name],
                    **managed_agent.agent_args or {},
                )
                agents.append(managed_instance)

            main_tools = await self._load_tools(self.config.tools)
            main_agent = ReActAgent(
                name=self.config.name,
                description=self.config.description,
                tools=main_tools,
                llm=self._get_model(self.config),
                system_prompt=self.config.instructions,
                can_handoff_to=managed_names,
                **self.config.agent_args or {},
            )
            agents.append(main_agent)

            self._agent = AgentWorkflow(agents=agents, root_agent=main_agent.name)

        else:
            imported_tools = await self._load_tools(self.config.tools)

            self._agent = ReActAgent(
                name=self.config.name,
                tools=imported_tools,
                llm=self._get_model(self.config),
                system_prompt=self.config.instructions,
                **self.config.agent_args or {},
            )

    async def run_async(self, prompt):
        result = await self._agent.run(prompt)
        return result

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._agent.tools
