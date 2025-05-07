from collections.abc import Generator, Sequence
from unittest.mock import AsyncMock, patch

import pytest
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
    SseServerParams as GoogleSseServerParameters,
)

from any_agent.config import AgentFramework, MCPSse, Tool
from any_agent.tools import _get_mcp_server


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

    google_sse_params.assert_called_once_with(  # type: ignore[attr-defined]
        url=mcp_sse_params_no_tools.url,
        headers=mcp_sse_params_no_tools.headers,
    )

    google_toolset.assert_called_once_with(  # type: ignore[attr-defined]
        connection_params=google_sse_params.return_value  # type: ignore[attr-defined]
    )

    google_toolset().load_tools.assert_called_once()  # type: ignore[operator]
