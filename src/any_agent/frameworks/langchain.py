import importlib
from typing import TYPE_CHECKING, Any, cast

from any_agent.config import AgentConfig, AgentFramework, Tool
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.logging import logger
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import wrap_tools

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.graph.graph import CompiledGraph

    from any_agent.tools.mcp import MCPServerBase

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike

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
        managed_agents: list[AgentConfig] | None = None,
    ):
        if not langchain_available:
            msg = "You need to `pip install 'any-agent[langchain]'` to use this agent"
            raise ImportError(msg)
        self.managed_agents = managed_agents
        self.config = config
        self._agent: CompiledGraph | None = None
        # Langgraph doesn't let you easily access what tools are loaded from the CompiledGraph,
        # so we'll store a list of them in this class
        self._tools: Sequence[Tool] = []
        self._mcp_servers: Sequence[MCPServerBase] | None = None

    def _get_model(self, agent_config: AgentConfig) -> str | LanguageModelLike:
        """Get the model configuration for a LangChain agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(importlib.import_module(module), class_name)

        return cast(
            str | LanguageModelLike,
            model_type(model=agent_config.model_id, **agent_config.model_args or {}),
        )

    async def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""
        if not self.managed_agents and not self.config.tools:
            self.config.tools = [
                search_web,
                visit_webpage,
            ]

        imported_tools, mcp_servers = await wrap_tools(
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
                managed_tools, managed_mcp_servers = await wrap_tools(
                    managed_agent.tools, agent_framework=AgentFramework.LANGCHAIN
                )
                managed_tools.extend(
                    [
                        tool
                        for mcp_server in managed_mcp_servers
                        for tool in mcp_server.tools
                    ],
                )
                name = managed_agent.name
                if not name or name == "any_agent":
                    logger.warning(
                        "Overriding name for managed_agent. Can't use the default.",
                    )
                    name = f"managed_agent_{n}"
                managed_names.append(name)

                managed_agent = create_react_agent(
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
            self._agent = workflow.compile()
            self._tools = imported_tools
        else:
            self._agent = create_react_agent(
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
        return await self._agent.ainvoke(inputs)  # type: ignore[union-attr]

    @property
    def tools(self) -> list[Tool]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return list(self._tools)
