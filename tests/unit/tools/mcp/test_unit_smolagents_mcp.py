# pylint: disable=unused-argument, unused-variable, attr-de
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSse, MCPStdio, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def smolagents_mcp_server(
    tools: Sequence[Tool],
) -> Generator[MagicMock]:
    with patch(
        "any_agent.tools.mcp.frameworks.smolagents.MCPClient"
    ) as mock_client_class:
        mock_client_class.return_value.__enter__.return_value = tools
        yield mock_client_class


@pytest.mark.asyncio
async def test_smolagents_mcp_sse_integration(
    mcp_sse_params_no_tools: MCPSse,
    smolagents_mcp_server: MagicMock,
) -> None:
    server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.SMOLAGENTS)

    await server._setup_tools()

    # Check that MCPClient was called with server parameters and adapter_kwargs
    smolagents_mcp_server.assert_called_once_with(
        {"url": mcp_sse_params_no_tools.url},
        adapter_kwargs={
            "client_session_timeout_seconds": mcp_sse_params_no_tools.client_session_timeout_seconds
        },
    )


@pytest.mark.asyncio
async def test_smolagents_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.SMOLAGENTS,
    )
    mocked_class = MagicMock()

    with (
        patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient", mocked_class),
    ):
        await mcp_server._setup_tools()
        assert mocked_class.call_args_list[0][0][0].env == {"FOO": "BAR"}


@pytest.mark.asyncio
async def test_smolagents_mcp_stdio_timeout() -> None:
    """Test that the timeout is properly passed to MCPClient for STDIO connections."""
    custom_timeout = 10.0
    mcp_server = _get_mcp_server(
        MCPStdio(
            command="test_command",
            args=[],
            env={},
            client_session_timeout_seconds=custom_timeout,
        ),
        AgentFramework.SMOLAGENTS,
    )
    mocked_class = MagicMock()

    with patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient", mocked_class):
        await mcp_server._setup_tools()
        call_args = mocked_class.call_args_list[0]
        assert (
            call_args[1]["adapter_kwargs"]["client_session_timeout_seconds"]
            == custom_timeout
        )


@pytest.mark.asyncio
async def test_smolagents_mcp_sse_timeout() -> None:
    """Test that the timeout is properly passed to MCPClient for SSE connections."""
    custom_timeout = 15.0
    mcp_sse_params = MCPSse(
        url="http://localhost:8000/sse",
        client_session_timeout_seconds=custom_timeout,
    )

    with patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient") as mock_client:
        mcp_server = _get_mcp_server(mcp_sse_params, AgentFramework.SMOLAGENTS)
        await mcp_server._setup_tools()

        mock_client.assert_called_once()
        call_args = mock_client.call_args_list[0]
        assert call_args[0][0] == {"url": mcp_sse_params.url}
        assert (
            call_args[1]["adapter_kwargs"]["client_session_timeout_seconds"]
            == custom_timeout
        )
