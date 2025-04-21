from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, assert_never

from any_agent.config import AgentConfig, AgentFramework, Tool, TracingConfig
from any_agent.logging import logger
from any_agent.tracing import setup_tracing


class AnyAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    trace_filepath: str | None = None

    # factory method
    @staticmethod
    def _get_agent_type_by_framework(
        framework_raw: AgentFramework | str,
    ) -> type[AnyAgent]:
        framework = AgentFramework.from_string(framework_raw)

        if framework is AgentFramework.SMOLAGENTS:
            from any_agent.frameworks.smolagents import SmolagentsAgent

            return SmolagentsAgent

        if framework is AgentFramework.LANGCHAIN:
            from any_agent.frameworks.langchain import LangchainAgent

            return LangchainAgent

        if framework is AgentFramework.OPENAI:
            from any_agent.frameworks.openai import OpenAIAgent

            return OpenAIAgent

        if framework is AgentFramework.LLAMA_INDEX:
            from any_agent.frameworks.llama_index import LlamaIndexAgent

            return LlamaIndexAgent

        if framework is AgentFramework.GOOGLE:
            from any_agent.frameworks.google import GoogleAgent

            return GoogleAgent

        if framework is AgentFramework.AGNO:
            from any_agent.frameworks.agno import AgnoAgent

            return AgnoAgent

        assert_never(framework)

    @classmethod
    def create(
        cls,
        agent_framework: AgentFramework | str,
        agent_config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ) -> AnyAgent:
        framework = AgentFramework.from_string(agent_framework)
        agent_cls = cls._get_agent_type_by_framework(agent_framework)
        agent = agent_cls(agent_config, managed_agents=managed_agents)
        if tracing is not None:
            # Agno not yet supported https://github.com/Arize-ai/openinference/issues/1302
            # Google ADK not yet supported https://github.com/Arize-ai/openinference/issues/1506
            if framework in (AgentFramework.AGNO, AgentFramework.GOOGLE):
                logger.warning(
                    "Tracing is not yet supported for AGNO and GOOGLE frameworks. "
                )
            else:
                agent.trace_filepath = setup_tracing(framework, tracing)
        asyncio.get_event_loop().run_until_complete(agent.load_agent())
        return agent

    @abstractmethod
    async def load_agent(self) -> None:
        """Load the agent instance."""

    def run(self, prompt: str) -> Any:
        """Run the agent with the given prompt."""
        return asyncio.get_event_loop().run_until_complete(self.run_async(prompt))

    @abstractmethod
    async def run_async(self, prompt: str) -> Any:
        """Run the agent asynchronously with the given prompt."""

    @property
    @abstractmethod
    def tools(self) -> list[Tool]:
        """Return the tools used by the agent.
        This property is read-only and cannot be modified.
        """

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
    ) -> None:
        msg = "Cannot instantiate the base class AnyAgent, please use the factory method 'AnyAgent.create'"
        raise NotImplementedError(msg)

    @property
    def agent(self) -> Any:
        """The underlying agent implementation from the framework.

        This property is intentionally restricted to maintain framework abstraction
        and prevent direct dependency on specific agent implementations.

        If you need functionality that relies on accessing the underlying agent:
        1. Consider if the functionality can be added to the AnyAgent interface
        2. Submit a GitHub issue describing your use case
        3. Contribute a PR implementing the needed functionality

        Raises:
            NotImplementedError: Always raised when this property is accessed

        """
        msg = "Cannot access the 'agent' property of AnyAgent, if you need to use functionality that relies on the underlying agent framework, please file a Github Issue or we welcome a PR to add the functionality to the AnyAgent class"
        raise NotImplementedError(msg)
