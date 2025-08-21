"""Simplified MCP client that handles all transport types and frameworks."""

import inspect
import os
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from datetime import timedelta
from textwrap import dedent
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from any_agent.config import (
    AgentFramework,
    MCPParams,
    MCPSse,
    MCPStdio,
    MCPStreamableHttp,
)

missing_mcp_error = None
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool as MCPTool
except ImportError as e:
    missing_mcp_error = e


class MCPClient(BaseModel):
    """Unified MCP client that handles all transport types and frameworks."""

    config: MCPParams
    framework: AgentFramework

    _session: ClientSession | None = PrivateAttr(default=None)
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)
    _client: Any | None = PrivateAttr(default=None)
    _get_session_id_callback: Callable[[], str | None] | None = PrivateAttr(
        default=None
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize the MCP client and check dependencies."""
        if missing_mcp_error:
            msg = "You need to `pip install 'any-agent[mcp]'` to use MCP."
            raise ImportError(msg) from missing_mcp_error

    async def connect(self) -> None:
        """Connect using the appropriate transport type."""
        if isinstance(self.config, MCPStdio):
            server_params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env={**os.environ},
            )
            self._client = stdio_client(server_params)
            read, write = await self._exit_stack.enter_async_context(self._client)
        elif isinstance(self.config, MCPSse):
            self._client = sse_client(
                url=self.config.url,
                headers=dict(self.config.headers or {}),
            )
            read, write = await self._exit_stack.enter_async_context(self._client)
        elif isinstance(self.config, MCPStreamableHttp):
            self._client = streamablehttp_client(
                url=self.config.url,
                headers=dict(self.config.headers or {}),
            )
            transport = await self._exit_stack.enter_async_context(self._client)
            read, write, self._get_session_id_callback = transport
        else:
            msg = f"Unsupported MCP config type: {type(self.config)}"
            raise ValueError(msg)

        # Create and initialize session (common for all transports)
        timeout = (
            timedelta(seconds=self.config.client_session_timeout_seconds)
            if self.config.client_session_timeout_seconds
            else None
        )
        client_session = ClientSession(read, write, timeout)
        self._session = await self._exit_stack.enter_async_context(client_session)
        await self._session.initialize()

    async def list_raw_tools(self) -> list[MCPTool]:
        """Get raw MCP tools from the server."""
        if not self._session:
            msg = "Not connected to MCP server. Call connect() first."
            raise ValueError(msg)

        available_tools = await self._session.list_tools()
        return self._filter_tools(available_tools.tools)

    async def list_tools(self) -> list[Callable[..., Any]]:
        """Get tools converted to callable functions that work with any framework."""
        raw_tools = await self.list_raw_tools()
        return self._convert_tools_to_callables(raw_tools)

    def _filter_tools(self, tools: Sequence[MCPTool]) -> list[MCPTool]:
        """Filter tools based on config."""
        requested_tools = list(self.config.tools or [])
        if not requested_tools:
            return list(tools)

        name_to_tool = {tool.name: tool for tool in tools}
        missing_tools = [name for name in requested_tools if name not in name_to_tool]
        if missing_tools:
            error_message = dedent(
                f"""Could not find all requested tools in the MCP server:
                Requested ({len(requested_tools)}): {requested_tools}
                Available ({len(name_to_tool)}): {list(name_to_tool.keys())}
                Missing: {missing_tools}
                """
            )
            raise ValueError(error_message)

        return [name_to_tool[name] for name in requested_tools]

    def _convert_tools_to_callables(
        self, tools: list[MCPTool]
    ) -> list[Callable[..., Any]]:
        """Convert MCP tools to callable functions that work with any framework."""
        if not self._session:
            msg = "Session not available for tool conversion"
            raise ValueError(msg)
        tool_functions = []
        for tool in tools:
            tool_func = self._create_tool_function(tool)
            tool_functions.append(tool_func)
        return tool_functions

    def _create_tool_function(self, tool: MCPTool) -> Callable[..., Any]:
        """Create a properly typed function for an MCP tool."""
        name = tool.name
        description = tool.description or f"MCP tool: {name}"
        input_schema = tool.inputSchema

        # Extract parameters from schema
        parameters = []
        annotations = {}
        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for param_name, param_info in properties.items():
                # Use the improved schema conversion
                base_param_type = self._json_schema_to_python_type(param_info)

                if param_name not in required:
                    # For optional parameters, use Optional[T] for better framework compatibility
                    # Note: We use Optional instead of X | Y syntax because some frameworks
                    # (like Google) don't handle the union syntax properly
                    optional_param_type: Any = Optional[base_param_type]  # noqa: UP045
                    annotations[param_name] = optional_param_type
                    param = inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=optional_param_type,
                    )
                else:
                    required_param_type: Any = base_param_type
                    annotations[param_name] = required_param_type
                    param = inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=required_param_type,
                    )
                parameters.append(param)

        # Create signature and enhanced docstring
        signature = inspect.Signature(parameters, return_annotation=str)
        enhanced_description = self._create_enhanced_description(
            description, input_schema
        )

        # Create the actual function
        async def mcp_tool_function(**kwargs: Any) -> str:
            """Dynamically created MCP tool function."""
            try:
                if not self._session:
                    return f"Error: MCP session not available for tool {name}"
                result = await self._session.call_tool(name, kwargs)
                if hasattr(result, "content") and result.content:
                    if hasattr(result.content[0], "text"):
                        return result.content[0].text
                    return str(result.content[0])
                return str(result)
            except Exception as e:
                return f"Error calling MCP tool {name}: {e!s}"

        # Set function metadata
        mcp_tool_function.__name__ = name
        mcp_tool_function.__doc__ = enhanced_description
        mcp_tool_function.__signature__ = signature  # type: ignore[attr-defined]
        mcp_tool_function.__annotations__ = {**annotations, "return": str}

        return mcp_tool_function

    TYPE_MAPPING: ClassVar[dict[str, type]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _json_schema_to_python_type(self, schema: dict[str, Any]) -> type:
        """Convert JSON schema to Python type using robust conversion."""
        schema_type = schema.get("type", "string")
        return self.TYPE_MAPPING.get(schema_type, str)

    def _create_enhanced_description(self, description: str, input_schema: Any) -> str:
        """Create enhanced docstring with parameter descriptions."""
        enhanced_description = description
        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
            if properties:
                param_descriptions = []
                for param_name, param_info in properties.items():
                    param_desc = param_info.get(
                        "description", f"Parameter {param_name}"
                    )
                    param_descriptions.append(f"    {param_name}: {param_desc}")

                if param_descriptions:
                    enhanced_description += "\n\nArgs:\n" + "\n".join(
                        param_descriptions
                    )
        return enhanced_description

    async def disconnect(self) -> None:
        """Clean up resources."""
        await self._exit_stack.aclose()
        self._session = None
        self._client = None
