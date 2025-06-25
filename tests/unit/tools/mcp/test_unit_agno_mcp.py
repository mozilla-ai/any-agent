from collections.abc import Generator, Sequence
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.tools.mcp import MCPTools as AgnoMCPTools

from any_agent.config import AgentFramework, MCPParams, MCPSse, MCPStdio, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def agno_mcp_tools() -> Generator[AgnoMCPTools, None, None]:
    with patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools") as mock_mcp_tools:
        yield mock_mcp_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session",
)
async def test_agno_mcp_sse_integration(
    mcp_sse_params_with_tools: MCPSse,
    session: Any,
    tools: Sequence[Tool],
    agno_mcp_tools: AgnoMCPTools,
) -> None:
    mcp_server = _get_mcp_server(mcp_sse_params_with_tools, AgentFramework.AGNO)

    await mcp_server._setup_tools()

    session.initialize.assert_called_once()

    agno_mcp_tools.assert_called_once_with(session=session, include_tools=tools)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session",
)
async def test_agno_mcp_no_tools(
    mcp_params_no_tools: MCPParams,
    agno_mcp_tools: AgnoMCPTools,
) -> None:
    """Regression test:"""
    mcp_server = _get_mcp_server(mcp_params_no_tools, AgentFramework.AGNO)

    await mcp_server._setup_tools()

    assert agno_mcp_tools.call_args_list[0].kwargs["include_tools"] is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_agno_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.AGNO,
    )
    mocked_class = MagicMock()
    mocked_cm = AsyncMock()
    mocked_cm.__aenter__.return_value = "foo"
    mocked_class.return_value = mocked_cm

    with patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools", mocked_class):
        await mcp_server._setup_tools()
        assert mocked_class.call_args_list[0][1]["env"] == {"FOO": "BAR"}


@pytest.mark.asyncio
async def test_agno_client_session_timeout_passed() -> None:
    """Test that client_session_timeout_seconds parameter is properly passed to AgnoMCPTools (STDIO only)."""
    custom_timeout = 15
    stdio_params = MCPStdio(
        command="echo",
        args=["test"],
        client_session_timeout_seconds=custom_timeout,
    )
    sse_params = MCPSse(
        url="http://localhost:8000",
        client_session_timeout_seconds=custom_timeout,
    )
    # STDIO
    server = _get_mcp_server(stdio_params, AgentFramework.AGNO)
    with patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools") as mock_agno_tools:
        mock_tools_instance = AsyncMock()
        mock_tools_instance.__aenter__ = AsyncMock(return_value=mock_tools_instance)
        mock_tools_instance.__aexit__ = AsyncMock(return_value=None)
        mock_agno_tools.return_value = mock_tools_instance
        await server._setup_tools()
        mock_agno_tools.assert_called_once()
        call_args = mock_agno_tools.call_args
        assert call_args.kwargs["timeout_seconds"] == custom_timeout
    # SSE (check that timeout is passed to ClientSession)
    server = _get_mcp_server(sse_params, AgentFramework.AGNO)
    with (
        patch("any_agent.tools.mcp.frameworks.agno.sse_client") as sse_client_patch,
        patch("any_agent.tools.mcp.frameworks.agno.ClientSession") as mock_session,
        patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools") as mock_agno_tools,
    ):
        mock_sse_client = AsyncMock()
        mock_sse_client.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_sse_client.__aexit__ = AsyncMock(return_value=None)
        sse_client_patch.return_value = mock_sse_client

        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session_instance.initialize = AsyncMock()
        mock_session.return_value = mock_session_instance

        mock_tools_instance = AsyncMock()
        mock_tools_instance.__aenter__ = AsyncMock(return_value=mock_tools_instance)
        mock_tools_instance.__aexit__ = AsyncMock(return_value=None)
        mock_agno_tools.return_value = mock_tools_instance

        await server._setup_tools()
        mock_session.assert_called_once()
        # Check that the timeout was passed as read_timeout_seconds
        call_args = mock_session.call_args
        assert call_args.kwargs["read_timeout_seconds"] == timedelta(
            seconds=custom_timeout
        )
