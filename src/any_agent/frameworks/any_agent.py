from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, assert_never, overload

from opentelemetry import trace as otel_trace

from any_agent.config import (
    AgentConfig,
    AgentFramework,
    Tool,
)
from any_agent.logging import logger
from any_agent.tools.wrappers import _wrap_tools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.exporter import SCOPE_NAME
from any_agent.tracing.instrumentation import (
    _get_instrumentor_by_framework,
)
from any_agent.utils import run_async_in_sync

if TYPE_CHECKING:
    from collections.abc import Sequence

    import uvicorn
    from opentelemetry.trace import Tracer
    from pydantic import BaseModel

    from any_agent.serving import A2AServingConfig, MCPServingConfig
    from any_agent.tools.mcp.mcp_server import _MCPServerBase


class AgentRunError(Exception):
    """Error that wraps underlying framework specific errors and carries spans."""

    _trace: AgentTrace
    _original_exception: Exception

    def __init__(self, trace: AgentTrace, original_exception: Exception):
        self._trace = trace
        self._original_exception = original_exception
        # Set the exception message to be the original exception's message
        super().__init__(str(original_exception))

    @property
    def trace(self) -> AgentTrace:
        return self._trace

    @property
    def original_exception(self) -> Exception:
        return self._original_exception

    def __str__(self) -> str:
        """Return the string representation of the original exception."""
        return str(self._original_exception)

    def __repr__(self) -> str:
        """Return the detailed representation of the AgentRunError."""
        return f"AgentRunError({self._original_exception!r})"


class AnyAgent(ABC):
    """Base abstract class for all agent implementations.

    This provides a unified interface for different agent frameworks.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        self._mcp_servers: list[_MCPServerBase[Any]] = []
        self._tools: list[Any] = []

        self._instrumentor = _get_instrumentor_by_framework(self.framework)
        self._tracer: Tracer = otel_trace.get_tracer(SCOPE_NAME)

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
        return run_async_in_sync(
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
        return run_async_in_sync(self.run_async(prompt, **kwargs))

    async def run_async(
        self, prompt: str, instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        """Run the agent asynchronously with the given prompt.

        Args:
            prompt: The user prompt to be passed to the agent.
            instrument: Whether to instrument the underlying framework
                to generate LLM Calls and Tool Execution Spans.

                If `False` the returned `AgentTrace` will only
                contain a single `invoke_agent` span.

            kwargs: Will be passed to the underlying runner used
                by the framework.

        Returns:
            The `AgentTrace` containing information about the
                steps taken by the agent.

        """
        trace = AgentTrace()
        trace_id: int
        instrumentation_enabled = instrument and self._instrumentor is not None

        # This design is so that we only catch exceptions thrown by _run_async. All other exceptions will not be caught.
        try:
            with self._tracer.start_as_current_span(
                f"invoke_agent [{self.config.name}]"
            ) as invoke_span:
                if instrumentation_enabled:
                    trace_id = invoke_span.get_span_context().trace_id
                    async with self._lock:
                        # We check the locked `_running_traces` inside `instrument`.
                        # If there is more than 1 entry in `running_traces`, it means that the agent has
                        # already being instrumented so we won't instrument it again.
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
        except Exception as e:
            # Clean up instrumentation if it was enabled
            if instrumentation_enabled:
                async with self._lock:
                    self._instrumentor.uninstrument(self)  # type: ignore[arg-type]
                    # Get the instrumented trace if available, otherwise use the original trace
                    instrumented_trace = self._running_traces.pop(trace_id)
                    if instrumented_trace is not None:
                        trace = instrumented_trace
            trace.add_span(invoke_span)
            raise AgentRunError(trace, e) from e

        if instrumentation_enabled:
            async with self._lock:
                self._instrumentor.uninstrument(self)  # type: ignore[arg-type]
                trace = self._running_traces.pop(trace_id)

        trace.add_span(invoke_span)
        trace.final_output = final_output
        return trace

    def _serve_a2a(self, serving_config: A2AServingConfig | None) -> None:
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

    def _serve_mcp(self, serving_config: MCPServingConfig) -> None:
        from any_agent.serving import (
            serve_mcp,
        )

        serve_mcp(
            self,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    @overload
    def serve(self, serving_config: MCPServingConfig) -> None: ...

    @overload
    def serve(self, serving_config: A2AServingConfig | None = None) -> None: ...

    def serve(
        self, serving_config: MCPServingConfig | A2AServingConfig | None = None
    ) -> None:
        """Serve this agent using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig or MCPServingConfig.

        Raises:
            ImportError: If the `a2a` dependencies are not installed and an `A2AServingConfig` is used.

        Example:
            ```
            agent = AnyAgent.create("tinyagent", AgentConfig(...))
            config = A2AServingConfig(port=8080, endpoint="/my-agent")
            agent.serve(config)
            ```

        """
        from any_agent.serving import MCPServingConfig

        if isinstance(serving_config, MCPServingConfig):
            self._serve_mcp(serving_config)
        else:
            self._serve_a2a(serving_config)

    async def _serve_a2a_async(
        self, serving_config: A2AServingConfig | None
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]:
        from any_agent.serving import (
            A2AServingConfig,
            _get_a2a_app_async,
            serve_a2a_async,
        )

        if serving_config is None:
            serving_config = A2AServingConfig()

        app = await _get_a2a_app_async(self, serving_config=serving_config)

        return await serve_a2a_async(
            app,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    async def _serve_mcp_async(
        self, serving_config: MCPServingConfig
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]:
        from any_agent.serving import serve_mcp_async

        return await serve_mcp_async(
            self,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    @overload
    async def serve_async(
        self, serving_config: MCPServingConfig
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]: ...

    @overload
    async def serve_async(
        self, serving_config: A2AServingConfig | None = None
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]: ...

    async def serve_async(
        self, serving_config: MCPServingConfig | A2AServingConfig | None = None
    ) -> tuple[asyncio.Task[Any], uvicorn.Server]:
        """Serve this agent asynchronously using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig or MCPServingConfig.

        Raises:
            ImportError: If the `a2a` dependencies are not installed and an `A2AServingConfig` is used.

        Example:
            ```
            agent = await AnyAgent.create_async("tinyagent", AgentConfig(...))
            config = MCPServingConfig(port=8080)
            task, server = await agent.serve_async(config)
            try:
                # Server is running
                await asyncio.sleep(10)
            finally:
                server.should_exit = True
                await task
            ```

        """
        from any_agent.serving import MCPServingConfig

        if isinstance(serving_config, MCPServingConfig):
            return await self._serve_mcp_async(serving_config)
        return await self._serve_a2a_async(serving_config)

    def _recreate_with_config(self, new_config: AgentConfig) -> AnyAgent:
        """Create a new agent instance with the given config, preserving MCP servers and tools.

        This method creates a new agent with the modified configuration while transferring
        the MCP servers and tools from the current agent to avoid recreating them, but only
        if the tools configuration hasn't changed.

        Args:
            new_config: The new configuration to use for the recreated agent.

        Returns:
            A new agent instance with the modified config and transferred state (if tools unchanged)
            or a completely new agent (if tools changed).

        """
        return run_async_in_sync(self._recreate_with_config_async(new_config))

    async def _recreate_with_config_async(self, new_config: AgentConfig) -> AnyAgent:
        """Async version of _recreate_with_config.

        This method creates a new agent with the modified configuration while transferring
        the MCP servers and tools from the current agent to avoid recreating them, but only
        if the tools configuration hasn't changed.

        Args:
            new_config: The new configuration to use for the recreated agent.

        Returns:
            A new agent instance with the modified config and transferred state (if tools unchanged)
            or a completely new agent (if tools changed).

        """
        # Check if tools configuration has changed
        if self.config.tools != new_config.tools:
            # Tools have changed, so we need to recreate everything from scratch
            logger.info(
                "Tools have changed, so we need to recreate everything from scratch"
            )
            return await self.create_async(self.framework, new_config)

        # Tools haven't changed, so we can safely preserve MCP servers and tools
        # Create the new agent with the modified config
        # Don't use AnyAgent.create_async(), because it will recreate the MCP servers and tools
        new_agent = self.__class__(new_config)

        # Transfer MCP servers and tools from the original agent to avoid recreating them
        new_agent._mcp_servers = self._mcp_servers
        new_agent._tools = self._tools

        # Load the agent with the existing MCP servers and tools
        await new_agent._load_agent()

        return new_agent

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
