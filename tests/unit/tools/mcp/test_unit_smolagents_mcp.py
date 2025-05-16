# pylint: disable=unused-argument, unused-variable, attr-de
from collections.abc import Generator, Sequence
from unittest.mock import MagicMock, patch

import pytest
from smolagents.mcp_client import MCPClient

from any_agent.config import AgentFramework, MCPSse, MCPStdio, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def smolagents_mcp_server(
    tools: Sequence[Tool],
) -> Generator[MCPClient]:
    with patch(
        "any_agent.tools.mcp.frameworks.smolagents.MCPClient"
    ) as mock_client_class:
        mock_client_class.return_value.__enter__.return_value = tools
        yield mock_client_class


@pytest.mark.asyncio
async def test_smolagents_mcp_sse_integration(
    mcp_sse_params_no_tools: MCPSse,
    smolagents_mcp_server: MCPClient,
) -> None:
    server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.SMOLAGENTS)

    await server._setup_tools()

    smolagents_mcp_server.assert_called_once_with({"url": mcp_sse_params_no_tools.url})  # type: ignore[attr-defined]


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
