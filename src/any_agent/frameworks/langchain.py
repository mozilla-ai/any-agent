import importlib
from typing import Any, Optional, List

from any_agent.config import AgentFramework, AgentConfig
from any_agent.logging import logger
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools.wrappers import import_and_wrap_tools

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.graph.graph import CompiledGraph
    from langgraph_swarm import create_handoff_tool, create_swarm

    langchain_available = True
except ImportError:
    langchain_available = False


DEFAULT_MODEL_CLASS = "langchain_litellm.ChatLiteLLM"


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        if not langchain_available:
            raise ImportError(
                "You need to `pip install 'any-agent[langchain]'` to use this agent"
            )
        self.managed_agents = managed_agents
        self.config = config
        self._agent = None
        # Langgraph doesn't let you easily access what tools are loaded from the CompiledGraph,
        # so we'll store a list of them in this class
        self._tools = []
        self._mcp_servers = None

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a LangChain agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(importlib.import_module(module), class_name)

        return model_type(model=agent_config.model_id, **agent_config.model_args or {})

    async def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""

        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]

        imported_tools, mcp_servers = await import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.LANGCHAIN
        )
        self._mcp_servers = mcp_servers

        # Extract tools from MCP managers and add them to the imported_tools list
        for mcp_server in mcp_servers:
            imported_tools.extend(mcp_server.tools)

        if self.managed_agents:
            swarm = []
            managed_names = []
            for n, managed_agent in enumerate(self.managed_agents):
                managed_tools, managed_mcp_servers = await import_and_wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.LANGCHAIN
                )
                managed_tools.extend(
                    [
                        tool
                        for mcp_server in managed_mcp_servers
                        for tool in mcp_server.tools
                    ]
                )
                name = managed_agent.name
                if not name or name == "any_agent":
                    logger.warning(
                        "Overriding name for managed_agent. Can't use the default."
                    )
                    name = f"managed_agent_{n}"
                managed_names.append(name)

                managed_agent = create_react_agent(
                    name=name,
                    model=self._get_model(managed_agent),
                    tools=managed_tools
                    + [create_handoff_tool(agent_name=self.config.name)],
                    prompt=managed_agent.instructions,
                    **managed_agent.agent_args or {},
                )
                swarm.append(managed_agent)

            imported_tools += [
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
            swarm.append(main_agent)
            workflow = create_swarm(swarm, default_active_agent=self.config.name)
            self._agent: CompiledGraph = workflow.compile()
            self._tools = imported_tools
        else:
            self._agent: CompiledGraph = create_react_agent(
                name=self.config.name,
                model=self._get_model(self.config),
                tools=imported_tools,
                prompt=self.config.instructions,
                **self.config.agent_args or {},
            )
            self._tools = imported_tools

    async def run_async(self, prompt: str) -> Any:
        """Run the LangChain agent with the given prompt."""
        inputs = {"messages": [("user", prompt)]}
        result = await self._agent.ainvoke(inputs)
        return result

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._tools
