from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, assert_never
from uuid import uuid4

from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from any_agent.config import (
    AgentConfig,
    AgentFramework,
    Tool,
    TracingConfig,
)
from any_agent.tools.wrappers import _wrap_tools
from any_agent.tracing.exporter import _AnyAgentExporter
from any_agent.tracing.instrumentation import (
    _get_instrumentor_by_framework,
    _Instrumentor,
)
from any_agent.tracing.trace_provider import TRACE_PROVIDER
from any_agent.utils import run_async_in_sync

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    import uvicorn
    from pydantic import BaseModel

    from any_agent.serving.config import A2AServingConfig
    from any_agent.tools.mcp.mcp_server import _MCPServerBase
    from any_agent.tracing.agent_trace import AgentTrace


class AgentRunError(Exception):
    """Error that wraps underlying framework specific errors and carries spans."""

    _trace: AgentTrace

    def __init__(self, trace: AgentTrace):
        self._trace = trace

    @property
    def trace(self) -> AgentTrace:
        return self._trace


class AnyAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    def __init__(
        self,
        config: AgentConfig,
        tracing: TracingConfig | None = None,
    ):
        self.config = config

        self._mcp_servers: list[_MCPServerBase[Any]] = []
        self._tools: list[Any] = []

        # Tracing is enabled by default
        self._tracing_config: TracingConfig = tracing or TracingConfig()
        self._instrumentor: _Instrumentor | None = None
        self._exporter: _AnyAgentExporter | None = None
        self._setup_tracing()

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
        tracing: TracingConfig | None = None,
    ) -> AnyAgent:
        """Create an agent using the given framework and config."""
        return run_async_in_sync(
            cls.create_async(
                agent_framework=agent_framework,
                agent_config=agent_config,
                tracing=tracing,
            )
        )

    @classmethod
    async def create_async(
        cls,
        agent_framework: AgentFramework | str,
        agent_config: AgentConfig,
        tracing: TracingConfig | None = None,
    ) -> AnyAgent:
        """Create an agent using the given framework and config."""
        agent_cls = cls._get_agent_type_by_framework(agent_framework)
        agent = agent_cls(agent_config, tracing=tracing)
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

    def _setup_tracing(self) -> None:
        """Initialize the tracing for the agent."""
        self._trace_provider = TRACE_PROVIDER
        self._tracer = self._trace_provider.get_tracer("any_agent")
        self._exporter = _AnyAgentExporter(self._tracing_config)
        self._trace_provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._instrumentor = _get_instrumentor_by_framework(self.framework)
        self._instrumentor.instrument(tracer=self._tracer)

    def run(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run the agent with the given prompt."""
        return run_async_in_sync(self.run_async(prompt, **kwargs))

    async def run_async(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run the agent asynchronously with the given prompt."""
        run_id = str(uuid4())
        try:
            with self._tracer.start_as_current_span(
                f"invoke_agent [{self.config.name}]"
            ) as invoke_span:
                invoke_span.set_attributes(
                    {
                        "gen_ai.operation.name": "invoke_agent",
                        "gen_ai.agent.name": self.config.name,
                        "gen_ai.agent.description": self.config.description
                        or "No description.",
                        "gen_ai.request.model": self.config.model_id,
                        "gen_ai.request.id": run_id,
                    }
                )
                final_output = await self._run_async(prompt, **kwargs)
        except Exception as e:
            trace = self._exporter.pop_trace(run_id)  # type: ignore[union-attr]
            raise AgentRunError(trace) from e
        else:
            trace = self._exporter.pop_trace(run_id)  # type: ignore[union-attr]
            trace.final_output = final_output
            return trace

    def serve(self, serving_config: A2AServingConfig | None = None) -> None:
        """Serve this agent using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig.

        Raises:
            ImportError: If the `serving` dependencies are not installed.

        Example:
            agent = AnyAgent.create("tinyagent", AgentConfig(...))
            config = A2AServingConfig(port=8080, endpoint="/my-agent")
            agent.serve(config)

        """
        from any_agent.serving import A2AServingConfig, _get_a2a_app, serve_a2a

        if serving_config is None:
            serving_config = A2AServingConfig()

        app = _get_a2a_app(self, serving_config=serving_config)

        serve_a2a(
            app,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    async def serve_async(
        self, serving_config: A2AServingConfig | None = None
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]:
        """Serve this agent asynchronously using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig.

        Returns:
            A tuple containing:
            - asyncio.Task: The server task (keep a reference to prevent garbage collection)
            - uvicorn.Server: The server instance for controlling the server lifecycle

        Raises:
            ImportError: If the `serving` dependencies are not installed.

        Example:
            >>> agent = await AnyAgent.create_async("tinyagent", AgentConfig(...))
            >>> config = A2AServingConfig(port=8080)
            >>> task, server = await agent.serve_async(config)
            >>> try:
            ...     # Server is running
            ...     await asyncio.sleep(10)
            >>> finally:
            ...     server.should_exit = True
            ...     await task

        """
        from any_agent.serving import A2AServingConfig, _get_a2a_app, serve_a2a_async

        if serving_config is None:
            serving_config = A2AServingConfig()

        if not isinstance(serving_config, A2AServingConfig):
            msg = (
                f"serving_config must be an instance of A2AServingConfig, "
                f"got {serving_config.type}. "
                f"Currently only A2A serving is supported."
            )
            raise ValueError(msg)
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
    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
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

    def exit(self) -> None:
        """Exit the agent and clean up resources."""
        if self._instrumentor is not None:
            self._instrumentor.uninstrument()
        self._instrumentor = None
        self._exporter = None
        self._mcp_servers = []  # drop references to mcp servers so that they get garbage collected
