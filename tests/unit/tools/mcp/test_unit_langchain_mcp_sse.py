from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSse, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def load_mcp_tools(
    tools: Sequence[Tool],
) -> Generator[MagicMock]:
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
