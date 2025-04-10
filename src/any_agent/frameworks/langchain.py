import importlib
from typing import Any, Optional, List

from loguru import logger

from any_agent.config import AgentFramework, AgentConfig
from any_agent.tools.wrappers import import_and_wrap_tools
from .any_agent import AnyAgent

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.graph.graph import CompiledGraph

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
        self._tools = []
        self._load_agent()

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a LangChain agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(importlib.import_module(module), class_name)

        return model_type(model=agent_config.model_id, **agent_config.model_args or {})

    @logger.catch(reraise=True)
    def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""

        if not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]

        if self.managed_agents:
            raise NotImplementedError("langchain managed agents are not supported yet")

        imported_tools, mcp_managers = import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.LANGCHAIN
        )

        # Extract tools from MCP managers and add them to the imported_tools list
        for manager in mcp_managers:
            imported_tools.extend(manager.tools)

        model = self._get_model(self.config)

        self._agent: CompiledGraph = create_react_agent(
            model=model,
            tools=imported_tools,
            prompt=self.config.instructions,
            **self.config.agent_args or {},
        )
        # Langgraph doesn't let you easily access what tools are loaded from the CompiledGraph, so we'll store a list of them in this class
        self._tools = imported_tools

    @logger.catch(reraise=True)
    async def run_async(self, prompt: str) -> Any:
        """Run the LangChain agent with the given prompt."""
        inputs = {"messages": [("user", prompt)]}
        message = None
        for s in self._agent.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                logger.debug(message)
            else:
                message.pretty_print()
        return message

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._tools
