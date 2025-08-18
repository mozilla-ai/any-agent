from contextlib import suppress
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from mcp import StdioServerParameters
from pydantic import PrivateAttr

from any_agent.config import MCPSse, MCPStdio, MCPStreamableHttp
from any_agent.tools.mcp.mcp_client import MCPClient

# Check for Smolagents MCP dependencies
smolagents_mcp_available = False
with suppress(ImportError):
    from smolagents.mcp_client import MCPClient as SmolagentsMCPClientLib

    smolagents_mcp_available = True

if TYPE_CHECKING:
    from smolagents.tools import Tool as SmolagentsToolImport


class SmolagentsMCPClient(MCPClient):
    """Smolagents-specific MCP client that uses smolagents.mcp_client.MCPClient."""

    _smolagents_client: "SmolagentsMCPClientLib | None" = PrivateAttr(default=None)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize the Smolagents MCP client and check dependencies."""
        if not smolagents_mcp_available:
            msg = "You need to `pip install 'any-agent[mcp,smolagents]'` to use Smolagents MCP."
            raise ImportError(msg)

    async def connect(self) -> None:
        """Connect using smolagents MCP client."""
        if isinstance(self.config, MCPStdio):
            server_params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env=self.config.env or {},
            )
            adapter_kwargs = {}
            if self.config.client_session_timeout_seconds:
                adapter_kwargs["connect_timeout"] = (
                    self.config.client_session_timeout_seconds
                )

            self._smolagents_client = SmolagentsMCPClientLib(
                server_params, adapter_kwargs=adapter_kwargs
            )
        elif isinstance(self.config, MCPSse):
            sse_server_params: dict[str, str] = {
                "url": self.config.url,
                "transport": "sse",
            }
            adapter_kwargs = {}
            if self.config.client_session_timeout_seconds:
                adapter_kwargs["connect_timeout"] = (
                    self.config.client_session_timeout_seconds
                )

            self._smolagents_client = SmolagentsMCPClientLib(
                sse_server_params, adapter_kwargs=adapter_kwargs
            )
        elif isinstance(self.config, MCPStreamableHttp):
            http_server_params: dict[str, str] = {
                "url": self.config.url,
                "transport": "streamable-http",
            }
            adapter_kwargs = {}
            if self.config.client_session_timeout_seconds:
                adapter_kwargs["connect_timeout"] = (
                    self.config.client_session_timeout_seconds
                )

            self._smolagents_client = SmolagentsMCPClientLib(
                http_server_params, adapter_kwargs=adapter_kwargs
            )
        else:
            msg = f"Unsupported MCP config type: {type(self.config)}"
            raise ValueError(msg)

    async def list_tools(self) -> list[Any]:
        """Get tools as SmolagentsTool objects wrapped as callables."""
        if not self._smolagents_client:
            msg = "Not connected to MCP server. Call connect() first."
            raise ValueError(msg)

        smolagents_tools = self._smolagents_client.get_tools()
        return self._filter_smolagents_tools(smolagents_tools)

    def _filter_smolagents_tools(
        self, tools: list["SmolagentsToolImport"]
    ) -> list["SmolagentsToolImport"]:
        """Filter tools based on config."""
        requested_tools = list(self.config.tools or [])
        if not requested_tools:
            return tools

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

    async def disconnect(self) -> None:
        """Clean up resources."""
        if self._smolagents_client:
            self._smolagents_client.disconnect()
        self._smolagents_client = None
