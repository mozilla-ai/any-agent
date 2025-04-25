# pylint: disable=unused-argument, unused-variable
import shutil
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Tool as MCPTool

from any_agent.config import AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.mark.asyncio
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
    mock_stdio_client: Any,
) -> None:
    tee = shutil.which("tee") or ""
    assert tee, "tee not found"
    # Mocking the command part of stdio is really tricky so instead we'll use
    # a real command that should be available on all systems (this is what openai-agents does too)
    mock_stdio_params = MCPStdioParams(
        command=tee,
        args=[],
        tools=["write_file", "read_file"],
    )

    mock_tool_list = MagicMock()
    mock_tool_list.tools = [
        MCPTool(name="write_file", inputSchema={"type": "string", "properties": {}}),
        MCPTool(name="read_file", inputSchema={"type": "string", "properties": {}}),
        MCPTool(name="other_tool", inputSchema={"type": "string", "properties": {}}),
    ]
    mock_list_tools.return_value = mock_tool_list
    server = _get_mcp_server(mock_stdio_params, agent_framework)
    await server._setup_tools()
    if agent_framework == AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == {"write_file", "read_file"}  # type: ignore[union-attr]
    else:
        assert len(server.tools) == 2  # ignore[arg-type]


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    echo_sse_server: Any,
    agent_framework: AgentFramework,
) -> None:
    mock_sse_params = MCPSseParams(
        url=echo_sse_server["url"], tools=["say_hi", "say_bye"]
    )

    server = _get_mcp_server(mock_sse_params, agent_framework)
    await server._setup_tools()
    if agent_framework == AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == {"say_hi", "say_bye"}  # type: ignore[union-attr]
    else:
        assert len(server.tools) == 2  # ignore[arg-type]
