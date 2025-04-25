# pylint: disable=unused-argument, unused-variable
import shutil
from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.fixture
def command() -> str:
    # Mocking the command part of stdio is really tricky so instead we'll use
    # a real command that should be available on all systems (this is what openai-agents does too)
    tee = shutil.which("tee") or ""
    assert tee, "tee not found"
    return tee


@pytest.fixture
def stdio_params(command: str) -> MCPStdioParams:
    return MCPStdioParams(
        command=command,
        args=[],
        tools=["write_file", "read_file"],
    )


@pytest.fixture
def tools() -> list[MCPTool]:
    return [
        MCPTool(name="write_file", inputSchema={"type": "string", "properties": {}}),
        MCPTool(name="read_file", inputSchema={"type": "string", "properties": {}}),
        MCPTool(name="other_tool", inputSchema={"type": "string", "properties": {}}),
    ]


@pytest.fixture
def _patch_client_session() -> Generator[ClientSession]:
    with patch("mcp.client.session.ClientSession") as mock_client_session:
        mock_client_session.initialize = AsyncMock()

        mock_tool_list = MagicMock()
        mock_tool_list.tools = [
            MCPTool(
                name="write_file", inputSchema={"type": "string", "properties": {}}
            ),
            MCPTool(name="read_file", inputSchema={"type": "string", "properties": {}}),
            MCPTool(
                name="other_tool", inputSchema={"type": "string", "properties": {}}
            ),
        ]

        mock_client_session.list_tools.return_value = mock_tool_list
        yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("_patch_stdio_client")
@patch(
    "mcp.client.session.ClientSession.initialize",
    new_callable=AsyncMock,
    return_value=None,
)
@patch("mcp.client.session.ClientSession.list_tools")
async def test_stdio_tool_filtering(
    mock_list_tools: AsyncMock,
    mock_initialize: AsyncMock,
    agent_framework: AgentFramework,
    stdio_params: MCPStdioParams,
    tools: Sequence[MCPTool],
) -> None:
    mock_tool_list = MagicMock()
    mock_tool_list.tools = tools
    mock_list_tools.return_value = mock_tool_list
    server = _get_mcp_server(stdio_params, agent_framework)
    await server._setup_tools()
    if agent_framework == AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == {"write_file", "read_file"}  # type: ignore[union-attr]
    else:
        assert len(server.tools) == 2  # ignore[arg-type]


@pytest.fixture
def sse_params(
    echo_sse_server: Any,
) -> MCPSseParams:
    return MCPSseParams(url=echo_sse_server["url"], tools=["say_hi", "say_bye"])


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    agent_framework: AgentFramework,
    sse_params: MCPSseParams,
) -> None:
    server = _get_mcp_server(sse_params, agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == {"say_hi", "say_bye"}  # type: ignore[union-attr]
    else:
        assert len(server.tools) == 2  # ignore[arg-type]
