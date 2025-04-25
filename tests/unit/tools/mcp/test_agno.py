from collections.abc import Sequence
from typing import Any

import pytest
from agno.tools.mcp import MCPTools as AgnoMCPTools

from any_agent.config import AgentFramework, MCPSseParams, Tool
from any_agent.tools import get_mcp_server


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session",
    "agno_mcp_tools",
    "_path_client_session",
)
async def test_agno_mcp_sse_tools_loaded(
    mcp_sse_params_with_tools: MCPSseParams,
    agno_mcp_tool_instance: AgnoMCPTools,
    tools: Sequence[Tool],
) -> None:
    mcp_server = get_mcp_server(mcp_sse_params_with_tools, AgentFramework.AGNO)

    await mcp_server.setup_tools()

    assert mcp_server.server == agno_mcp_tool_instance  # type: ignore[union-attr]
    assert mcp_server.tools == [tools]


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session",
)
async def test_agno_mcp_sse_integration(
    mcp_sse_params_with_tools: MCPSseParams,
    session: Any,
    tools: Sequence[Tool],
    agno_mcp_tools: AgnoMCPTools,
) -> None:
    mcp_server = get_mcp_server(mcp_sse_params_with_tools, AgentFramework.AGNO)

    await mcp_server.setup_tools()

    session.initialize.assert_called_once()

    agno_mcp_tools.assert_called_once_with(session=session, include_tools=tools)
