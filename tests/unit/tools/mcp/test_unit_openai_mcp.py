from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentFramework
from any_agent.config import AgentConfig, MCPSse, MCPStdio, Tool
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import _get_mcp_server

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse


@pytest.fixture
def openai_mcp_sse_server(
    tools: Sequence[Tool],
) -> Generator[OpenAIInternalMCPServerSse]:
    with patch(
        "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerSse",
    ) as mock_server:
        mock_server.return_value.list_tools = AsyncMock(return_value=tools)
        yield mock_server


@pytest.mark.usefixtures(
    "openai_mcp_sse_server",
)
def test_openai_mcpsse(
    mcp_sse_params_no_tools: MCPSse,
) -> None:
    """This is a test kept for legacy purposes."""
    agent_config = AgentConfig(
        model_id="mistral/mistral-small-latest", tools=[mcp_sse_params_no_tools]
    )

    agent = AnyAgent.create("openai", agent_config)

    servers = agent._mcp_servers
    assert servers

    server, *_ = agent._mcp_servers
    assert server.mcp_tool == mcp_sse_params_no_tools


@pytest.mark.asyncio
async def test_openai_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.OPENAI,
    )
    mocked_class = MagicMock()
    mocked_class.return_value = AsyncMock()

    with (
        patch(
            "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerStdio",
            mocked_class,
        ),
    ):
        await mcp_server._setup_tools()
        assert mocked_class.call_args_list[0][1]["params"]["env"] == {"FOO": "BAR"}


def test_openai_client_session_timeout_passed() -> None:
    """Test that client_session_timeout_seconds parameter is properly passed to OpenAI MCPServerStdio and MCPServerSse."""
    custom_timeout = 15.0
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
    server = _get_mcp_server(stdio_params, AgentFramework.OPENAI)
    with patch(
        "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerStdio"
    ) as mock_stdio:
        mock_server_instance = AsyncMock()
        mock_server_instance.__aenter__ = AsyncMock(return_value=mock_server_instance)
        mock_server_instance.__aexit__ = AsyncMock(return_value=None)
        mock_server_instance.list_tools = AsyncMock(return_value=[])
        mock_stdio.return_value = mock_server_instance
        import asyncio

        asyncio.run(server._setup_tools())
        mock_stdio.assert_called_once()
        call_args = mock_stdio.call_args
        assert call_args.kwargs["client_session_timeout_seconds"] == custom_timeout
    # SSE
    server = _get_mcp_server(sse_params, AgentFramework.OPENAI)
    with patch(
        "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerSse"
    ) as mock_sse:
        mock_server_instance = AsyncMock()
        mock_server_instance.__aenter__ = AsyncMock(return_value=mock_server_instance)
        mock_server_instance.__aexit__ = AsyncMock(return_value=None)
        mock_server_instance.list_tools = AsyncMock(return_value=[])
        mock_sse.return_value = mock_server_instance
        asyncio.run(server._setup_tools())
        mock_sse.assert_called_once()
        call_args = mock_sse.call_args
        assert call_args.kwargs["client_session_timeout_seconds"] == custom_timeout
