from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSse, MCPStdio, Tool
from any_agent.tools import _get_mcp_server

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
        SseServerParams as GoogleSseServerParameters,
    )


@pytest.fixture
def google_sse_params() -> Generator[GoogleSseServerParameters]:
    with patch(
        "any_agent.tools.mcp.frameworks.google.GoogleSseServerParameters"
    ) as mock_params:
        yield mock_params


@pytest.fixture
def google_toolset(tools: Sequence[Tool]) -> Generator[GoogleMCPToolset]:
    toolset = AsyncMock()
    toolset.load_tools.return_value = tools
    with patch(
        "any_agent.tools.mcp.frameworks.google.GoogleMCPToolset", return_value=toolset
    ) as mock_class:
        yield mock_class


@pytest.mark.asyncio
@pytest.mark.usefixtures("enter_context_with_transport_and_session")
async def test_google_mcp_sse_integration(
    google_toolset: GoogleMCPToolset,
    google_sse_params: GoogleSseServerParameters,
    mcp_sse_params_no_tools: MCPSse,
) -> None:
    mcp_server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.GOOGLE)
    await mcp_server._setup_tools()

    google_sse_params.assert_called_once_with(
        url=mcp_sse_params_no_tools.url,
        headers=dict(mcp_sse_params_no_tools.headers or {}),
        timeout=mcp_sse_params_no_tools.client_session_timeout_seconds,
        sse_read_timeout=mcp_sse_params_no_tools.client_session_timeout_seconds,
    )

    google_toolset.assert_called_once_with(  # type: ignore[attr-defined]
        connection_params=google_sse_params.return_value
    )

    google_toolset().get_tools.assert_called_once()  # type: ignore[operator]


@pytest.mark.asyncio
async def test_google_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.GOOGLE,
    )
    mocked_class = MagicMock()
    mocked_cm = AsyncMock()
    mocked_class.return_value = mocked_cm

    with patch("any_agent.tools.mcp.frameworks.google.GoogleMCPToolset", mocked_class):
        await mcp_server._setup_tools()
        # Check that the connection params include the timeout
        call_args = mocked_class.call_args_list[0][1]["connection_params"]
        assert call_args.server_params.env == {"FOO": "BAR"}
        assert call_args.timeout == 5  # Default timeout value


@pytest.mark.asyncio
async def test_google_mcp_stdio_timeout() -> None:
    """Test that the timeout is properly passed to StdioConnectionParams."""
    custom_timeout = 10.0
    mcp_server = _get_mcp_server(
        MCPStdio(
            command="print('Hello MCP')",
            args=[],
            env={"FOO": "BAR"},
            client_session_timeout_seconds=custom_timeout,
        ),
        AgentFramework.GOOGLE,
    )
    mocked_class = MagicMock()
    mocked_cm = AsyncMock()
    mocked_class.return_value = mocked_cm

    with patch("any_agent.tools.mcp.frameworks.google.GoogleMCPToolset", mocked_class):
        await mcp_server._setup_tools()
        # Check that the connection params include the custom timeout
        call_args = mocked_class.call_args_list[0][1]["connection_params"]
        assert call_args.server_params.env == {"FOO": "BAR"}
        assert call_args.timeout == custom_timeout


@pytest.mark.asyncio
async def test_google_mcp_sse_timeout() -> None:
    """Test that the timeout is properly passed to SSE connection parameters."""
    custom_timeout = 15.0
    mcp_sse_params = MCPSse(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
        client_session_timeout_seconds=custom_timeout,
    )

    with patch(
        "any_agent.tools.mcp.frameworks.google.GoogleSseServerParameters"
    ) as mock_sse_params:
        with patch(
            "any_agent.tools.mcp.frameworks.google.GoogleMCPToolset"
        ) as mock_toolset:
            # Mock the get_tools method to return an empty list
            mock_toolset_instance = AsyncMock()
            mock_toolset_instance.get_tools.return_value = []
            mock_toolset.return_value = mock_toolset_instance

            mcp_server = _get_mcp_server(mcp_sse_params, AgentFramework.GOOGLE)
            await mcp_server._setup_tools()

            mock_sse_params.assert_called_once_with(
                url=mcp_sse_params.url,
                headers=dict(mcp_sse_params.headers or {}),
                timeout=custom_timeout,
                sse_read_timeout=custom_timeout,
            )
