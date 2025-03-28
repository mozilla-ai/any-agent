import importlib
from typing import Any, Optional, List

from loguru import logger

from any_agent.config import AgentFramework, AgentConfig
from any_agent.tools.wrappers import import_and_wrap_tools
from .any_agent import AnyAgent

try:
    from llama_index.core.agent.workflow import ReActAgent

    llama_index_available = True
except ImportError:
    llama_index_available = False


DEFAULT_MODEL_CLASS = "openai.OpenAI"


class LlamaIndexAgent(AnyAgent):
    """LLamaIndex agent implementation that handles both loading and running."""

    def __init__(
        self, config: AgentConfig, managed_agents: Optional[list[AgentConfig]] = None
    ):
        self.managed_agents: Optional[list[AgentConfig]] = managed_agents
        self.config: AgentConfig = config
        self._agent = None
        self._load_agent()

    def _get_model(self, agent_config: AgentConfig):
        """Get the model configuration for a llama_index agent."""
        if not agent_config.model_type:
            agent_config.model_type = DEFAULT_MODEL_CLASS
        module, class_name = agent_config.model_type.split(".")
        model_type = getattr(
            importlib.import_module(f"llama_index.llms.{module}"), class_name
        )

        return model_type(model=agent_config.model_id, **agent_config.model_args or {})

    @logger.catch(reraise=True)
    def _load_agent(self) -> None:
        """Load the LLamaIndex agent with the given configuration."""
        if not llama_index_available:
            raise ImportError("You need to `pip install llama-index` to use this agent")

        if not self.config.tools:
            self.config.tools = [
                "any_agent.tools.search_web",
                "any_agent.tools.visit_webpage",
            ]

        if self.managed_agents:
            raise NotImplementedError(
                "llama-index managed agents are not supported yet"
            )

        imported_tools, mcp_managers = import_and_wrap_tools(
            self.config.tools, agent_framework=AgentFramework.LLAMAINDEX
        )

        # Extract tools from MCP managers and add them to the imported_tools list
        for manager in mcp_managers:
            imported_tools.extend(manager.tools)

        self._agent = ReActAgent(
            name=self.config.name,
            tools=imported_tools,
            llm=self._get_model(self.config),
            **self.config.agent_args or {},
        )

    async def _async_run(self, prompt):
        result = await self._agent.run(prompt)
        return result

    @logger.catch(reraise=True)
    def run(self, prompt: str) -> Any:
        """Run the LlamaIndex agent with the given prompt."""
        import asyncio

        return asyncio.run(self._async_run(prompt))

    @property
    def tools(self) -> List[str]:
        """
        Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """
        return self._agent.tools
