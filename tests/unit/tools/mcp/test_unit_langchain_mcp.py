from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSse, MCPStdio, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def load_mcp_tools(
    tools: Sequence[Tool],
) -> Generator[MagicMock, None, None]:
    with patch(
        "any_agent.tools.mcp.frameworks.langchain.load_mcp_tools"
    ) as mock_load_tools:
        mock_load_tools.return_value = tools
        yield mock_load_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session", "_path_client_session"
)
async def test_langchain_mcp_sse_integration(
    mcp_sse_params_no_tools: MCPSse,
    session: Any,
    load_mcp_tools: Any,
) -> None:
    server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.LANGCHAIN)

    await server._setup_tools()

    session.initialize.assert_called_once()

    load_mcp_tools.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_langchain_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.LANGCHAIN,
    )
    mocked_class = MagicMock()
    mocked_cm = AsyncMock()
    mocked_cm.__aenter__.return_value = ("stdio", "write")
    mocked_class.return_value = mocked_cm

    mocked_session = MagicMock()
    mocked_session_cm = AsyncMock()
    mocked_session.return_value = mocked_session_cm

    with (
        patch("any_agent.tools.mcp.frameworks.langchain.stdio_client", mocked_class),
        patch("any_agent.tools.mcp.frameworks.langchain.ClientSession", mocked_session),
        patch("any_agent.tools.mcp.frameworks.langchain.load_mcp_tools"),
    ):
        await mcp_server._setup_tools()
        assert mocked_class.call_args_list[0][0][0].env == {"FOO": "BAR"}


@pytest.mark.asyncio
async def test_langchain_client_session_timeout_passed() -> None:
    """Test that client_session_timeout_seconds parameter is properly passed to LangChain ClientSession (STDIO and SSE)."""
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
    server = _get_mcp_server(stdio_params, AgentFramework.LANGCHAIN)
    with patch(
        "any_agent.tools.mcp.frameworks.langchain.ClientSession"
    ) as mock_session:
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session_instance.initialize = AsyncMock()
        mock_session.return_value = mock_session_instance
        with patch(
            "any_agent.tools.mcp.frameworks.langchain.load_mcp_tools"
        ) as mock_load_tools:
            mock_load_tools.return_value = []
            with patch(
                "any_agent.tools.mcp.frameworks.langchain.stdio_client"
            ) as mock_client:
                mock_client.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock())
                )
                mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
                await server._setup_tools()
                mock_session.assert_called_once()
                call_args = mock_session.call_args
                assert (
                    call_args.kwargs["read_timeout_seconds"].total_seconds()
                    == custom_timeout
                )
    # SSE
    server = _get_mcp_server(sse_params, AgentFramework.LANGCHAIN)
    with patch(
        "any_agent.tools.mcp.frameworks.langchain.ClientSession"
    ) as mock_session:
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session_instance.initialize = AsyncMock()
        mock_session.return_value = mock_session_instance
        with patch(
            "any_agent.tools.mcp.frameworks.langchain.load_mcp_tools"
        ) as mock_load_tools:
            mock_load_tools.return_value = []
            with patch(
                "any_agent.tools.mcp.frameworks.langchain.sse_client"
            ) as mock_client:
                mock_client.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock())
                )
                mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
                await server._setup_tools()
                mock_session.assert_called_once()
                call_args = mock_session.call_args
                assert (
                    call_args.kwargs["read_timeout_seconds"].total_seconds()
                    == custom_timeout
                )
