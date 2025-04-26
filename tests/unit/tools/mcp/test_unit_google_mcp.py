from collections.abc import Sequence

import pytest
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
    SseServerParams as GoogleSseServerParameters,
)

from any_agent.config import AgentFramework, MCPSseParams, Tool
from any_agent.tools import _get_mcp_server


@pytest.mark.asyncio
@pytest.mark.usefixtures("enter_context_with_transport_and_session", "google_toolset")
async def test_google_mcp_sse_tools_loaded(
    tools: Sequence[Tool],
    mcp_sse_params_no_tools: MCPSseParams,
) -> None:
    mcp_server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.GOOGLE)
    await mcp_server._setup_tools()

    assert mcp_server.tools == tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("enter_context_with_transport_and_session")
async def test_google_mcp_sse_integration(
    google_toolset: GoogleMCPToolset,
    google_sse_params: GoogleSseServerParameters,
    mcp_sse_params_no_tools: MCPSseParams,
) -> None:
    mcp_server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.GOOGLE)
    await mcp_server._setup_tools()

    google_sse_params.assert_called_once_with(
        url=mcp_sse_params_no_tools.url,
        headers=mcp_sse_params_no_tools.headers,
    )

    google_toolset.assert_called_once_with(
        connection_params=google_sse_params.return_value
    )

    google_toolset().load_tools.assert_called_once()
