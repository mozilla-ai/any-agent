# pylint: disable=unused-argument, unused-variable
from collections.abc import Sequence

import pytest

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams, Tool
from any_agent.tools import MCPConnection, _get_mcp_server


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "_patch_stdio_client",
    "_patch_client_session_initialize",
    "_patch_client_session_list_tools",
)
async def test_stdio_tool_filtering(
    agent_framework: AgentFramework,
    stdio_params: MCPStdioParams,
    tools: Sequence[str],
) -> None:
    server = _get_mcp_server(stdio_params, agent_framework)
    await server._setup_tools()
    if agent_framework == AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == set(tools)  # type: ignore[union-attr]
    else:
        assert len(server.tools) == len(tools)  # ignore[arg-type]


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    agent_framework: AgentFramework,
    sse_params_echo_server: MCPSseParams,
    tools: Sequence[str],
) -> None:
    server = _get_mcp_server(sse_params_echo_server, agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == set(tools)  # type: ignore[union-attr]
    else:
        assert len(server.tools) == len(tools)  # ignore[arg-type]


@pytest.mark.asyncio
async def test_mcp_sse_tools_loaded(
    agent_framework: AgentFramework,
    mcp_sse_params_with_tools: MCPSseParams,
    mcp_connection: MCPConnection,
    tools: Sequence[Tool],
) -> None:
    if agent_framework is not AgentFramework.AGNO:
        pytest.skip("Only AGNO framework is supported for this test at the moment.")

    mcp_server = _get_mcp_server(mcp_sse_params_with_tools, AgentFramework.AGNO)

    await mcp_server._setup_tools(mcp_connection=mcp_connection)

    assert mcp_server.tools == list(tools)
