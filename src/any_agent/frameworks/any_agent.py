from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, assert_never

from any_agent.config import (
    AgentConfig,
    AgentFramework,
    ServingConfig,
    Tool,
)
from any_agent.tools.wrappers import _wrap_tools
from any_agent.tracing import TRACE_PROVIDER
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.instrumentation import (
    _get_instrumentor_by_framework,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import uvicorn
    from opentelemetry.trace import Tracer

    from any_agent.tools.mcp.mcp_server import _MCPServerBase


class AnyAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        self._mcp_servers: list[_MCPServerBase[Any]] = []
        self._tools: list[Any] = []

        self._instrumentor = _get_instrumentor_by_framework(self.framework)
        self._tracer: Tracer = TRACE_PROVIDER.get_tracer("any_agent")

        self._lock = asyncio.Lock()
        self._running_traces: dict[int, AgentTrace] = {}

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

        if framework is AgentFramework.TINYAGENT:
            from any_agent.frameworks.tinyagent import TinyAgent

            return TinyAgent

        assert_never(framework)

    @classmethod
    def create(
        cls,
        agent_framework: AgentFramework | str,
        agent_config: AgentConfig,
    ) -> AnyAgent:
        """Create an agent using the given framework and config."""
        return asyncio.get_event_loop().run_until_complete(
            cls.create_async(
                agent_framework=agent_framework,
                agent_config=agent_config,
            )
        )

    @classmethod
    async def create_async(
        cls,
        agent_framework: AgentFramework | str,
        agent_config: AgentConfig,
    ) -> AnyAgent:
        """Create an agent using the given framework and config."""
        agent_cls = cls._get_agent_type_by_framework(agent_framework)
        agent = agent_cls(agent_config)
        await agent._load_agent()
        return agent

    async def _load_tools(
        self, tools: Sequence[Tool]
    ) -> tuple[list[Any], list[_MCPServerBase[Any]]]:
        tools, mcp_servers = await _wrap_tools(tools, self.framework)
        # Add to agent so that it doesn't get garbage collected
        self._mcp_servers.extend(mcp_servers)
        for mcp_server in mcp_servers:
            tools.extend(mcp_server.tools)
        return tools, mcp_servers

    def run(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run the agent with the given prompt."""
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(prompt, **kwargs)
        )

    async def run_async(
        self, prompt: str, instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        """Run the agent asynchronously with the given prompt."""
        with self._tracer.start_as_current_span(
            f"invoke_agent [{self.config.name}]"
        ) as invoke_span:
            if instrument and self._instrumentor:
                trace_id = invoke_span.get_span_context().trace_id
                async with self._lock:
                    self._running_traces[trace_id] = AgentTrace()
                    self._instrumentor.instrument(
                        agent=self,  # type: ignore[arg-type]
                    )
            invoke_span.set_attributes(
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": self.config.name,
                    "gen_ai.agent.description": self.config.description
                    or "No description.",
                    "gen_ai.request.model": self.config.model_id,
                }
            )
            final_output = await self._run_async(prompt, **kwargs)

            trace = AgentTrace()
            if instrument and self._instrumentor:
                async with self._lock:
                    self._instrumentor.uninstrument(self)  # type: ignore[arg-type]
                    trace = self._running_traces.pop(trace_id)

        trace.add_span(invoke_span)
        trace.final_output = final_output

        return trace

    def serve(self, serving_config: ServingConfig | None = None) -> None:
        """Serve this agent using the Agent2Agent Protocol (A2A).

        Args:
            serving_config: See [ServingConfig][any_agent.config.ServingConfig].

        Raises:
            ImportError: If the `serving` dependencies are not installed.

        """
        from any_agent.serving import _get_a2a_app, serve_a2a

        if serving_config is None:
            serving_config = ServingConfig()
        app = _get_a2a_app(self, serving_config=serving_config)

        serve_a2a(
            app,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    async def serve_async(
        self, serving_config: ServingConfig | None = None
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]:
        """Serve this agent using the Agent2Agent Protocol (A2A).

        Args:
            serving_config: See [ServingConfig][any_agent.config.ServingConfig].

        Raises:
            ImportError: If the `serving` dependencies are not installed.

        """
        from any_agent.serving import _get_a2a_app, serve_a2a_async

        if serving_config is None:
            serving_config = ServingConfig()
        app = _get_a2a_app(self, serving_config=serving_config)

        return await serve_a2a_async(
            app,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    @abstractmethod
    async def _load_agent(self) -> None:
        """Load the agent instance."""

    @abstractmethod
    async def _run_async(self, prompt: str, **kwargs: Any) -> str:
        """To be implemented by each framework."""

    @property
    @abstractmethod
    def framework(self) -> AgentFramework:
        """The Agent Framework used."""

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
