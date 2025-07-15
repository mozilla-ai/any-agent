from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, assert_never, overload

from opentelemetry import trace as otel_trace

from any_agent.callbacks.context import Context
from any_agent.callbacks.wrappers import (
    _get_wrapper_by_framework,
)
from any_agent.config import (
    AgentConfig,
    AgentFramework,
    Tool,
)
from any_agent.logging import logger
from any_agent.tools.wrappers import _wrap_tools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.attributes import GenAI
from any_agent.utils import run_async_in_sync

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import Tracer
    from pydantic import BaseModel

    from any_agent.serving import A2AServingConfig, MCPServingConfig, ServerHandle
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

        self._add_span_callbacks()
        self._wrapper = _get_wrapper_by_framework(self.framework)

        self._tracer: Tracer = otel_trace.get_tracer("any_agent")

        self._lock = asyncio.Lock()
        self._callback_contexts: dict[int, Context] = {}

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

    async def run_async(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run the agent asynchronously with the given prompt.

        Args:
            prompt: The user prompt to be passed to the agent.

            kwargs: Will be passed to the underlying runner used
                by the framework.

        Returns:
            The `AgentTrace` containing information about the
                steps taken by the agent.

        """
        trace = AgentTrace()
        trace_id: int

        # This design is so that we only catch exceptions thrown by _run_async. All other exceptions will not be caught.
        try:
            with self._tracer.start_as_current_span(
                f"invoke_agent [{self.config.name}]"
            ) as invoke_span:
                async with self._lock:
                    trace_id = invoke_span.get_span_context().trace_id
                    self._wrapper.callback_context[trace_id] = Context(
                        current_span=invoke_span,
                        trace=AgentTrace(),
                        tracer=self._tracer,
                        shared={},
                    )

                    if len(self._wrapper.callback_context) == 1:
                        # If there is more than 1 entry in `callback_context`, it means that the agent has
                        # already being wrapped so we won't wrap it again.
                        await self._wrapper.wrap(
                            agent=self,  # type: ignore[arg-type]
                        )

                invoke_span.set_attributes(
                    {
                        GenAI.OPERATION_NAME: "invoke_agent",
                        GenAI.AGENT_NAME: self.config.name,
                        GenAI.AGENT_DESCRIPTION: self.config.description
                        or "No description.",
                        GenAI.REQUEST_MODEL: self.config.model_id,
                    }
                )

                final_output = await self._run_async(prompt, **kwargs)
        except Exception as e:
            async with self._lock:
                if len(self._wrapper.callback_context) == 1:
                    await self._wrapper.unwrap(self)  # type: ignore[arg-type]
                if wrapped_context := self._wrapper.callback_context.pop(
                    trace_id, None
                ):
                    trace = wrapped_context.trace
            trace.add_span(invoke_span)
            raise AgentRunError(trace, e) from e

        async with self._lock:
            if len(self._wrapper.callback_context) == 1:
                await self._wrapper.unwrap(self)  # type: ignore[arg-type]
            if wrapped_context := self._wrapper.callback_context.pop(trace_id, None):
                trace = wrapped_context.trace

        trace.add_span(invoke_span)
        trace.final_output = final_output
        return trace

    async def _serve_a2a_async(
        self, serving_config: A2AServingConfig | None
    ) -> ServerHandle:
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

    async def _serve_mcp_async(self, serving_config: MCPServingConfig) -> ServerHandle:
        from any_agent.serving import serve_mcp_async

        return await serve_mcp_async(
            self,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    @overload
    async def serve_async(self, serving_config: MCPServingConfig) -> ServerHandle: ...

    @overload
    async def serve_async(
        self, serving_config: A2AServingConfig | None = None
    ) -> ServerHandle: ...

    async def serve_async(
        self, serving_config: MCPServingConfig | A2AServingConfig | None = None
    ) -> ServerHandle:
        """Serve this agent asynchronously using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig or MCPServingConfig.

        Returns:
            A ServerHandle instance that provides methods for managing the server lifecycle.

        Raises:
            ImportError: If the `a2a` dependencies are not installed and an `A2AServingConfig` is used.

        Example:
            ```
            agent = await AnyAgent.create_async("tinyagent", AgentConfig(...))
            config = MCPServingConfig(port=8080)
            server_handle = await agent.serve_async(config)
            try:
                # Server is running
                await asyncio.sleep(10)
            finally:
                await server_handle.shutdown()
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

    def _add_span_callbacks(self) -> None:
        if self.config.callbacks is None:
            return

        from any_agent.callbacks.span_end import SpanEndCallback
        from any_agent.callbacks.span_generation import (
            SpanGeneration,
            _get_span_generation_callback,
        )

        if not any(isinstance(c, SpanGeneration) for c in self.config.callbacks):
            self.config.callbacks.insert(
                0, _get_span_generation_callback(self.framework)
            )
        if not any(isinstance(c, SpanEndCallback) for c in self.config.callbacks):
            self.config.callbacks.append(SpanEndCallback())

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
