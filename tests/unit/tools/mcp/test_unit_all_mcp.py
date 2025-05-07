# pylint: disable=unused-argument, unused-variable
from collections.abc import Sequence
from typing import Any

import pytest

from any_agent.config import AgentFramework, MCPParams, MCPSse, Tool
from any_agent.tools import _get_mcp_server, _MCPConnection


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    agent_framework: AgentFramework,
    sse_params_echo_server: MCPSse,
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
async def test_mcp_tools_loaded(
    agent_framework: AgentFramework,
    mcp_params: MCPParams,
    mcp_connection: _MCPConnection[Any],
    tools: Sequence[Tool],
) -> None:
    mcp_server = _get_mcp_server(mcp_params, agent_framework)

    await mcp_server._setup_tools(mcp_connection=mcp_connection)

    assert mcp_server.tools == list(tools)
