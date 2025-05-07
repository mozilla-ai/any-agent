# pylint: disable=unused-argument, unused-variable, attr-de
from collections.abc import Generator, Sequence
from unittest.mock import patch

import pytest
from smolagents.mcp_client import MCPClient

from any_agent.config import AgentFramework, MCPSse, Tool
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
