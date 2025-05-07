from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import patch

import pytest
from agno.tools.mcp import MCPTools as AgnoMCPTools

from any_agent.config import AgentFramework, MCPSse, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def agno_mcp_tools() -> Generator[AgnoMCPTools]:
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
