# pylint: disable=unused-argument, unused-variable
import shutil
from collections.abc import Generator, Sequence
from typing import Any, Protocol
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
def tools() -> list[str]:
    return ["write_file", "read_file", "other_tool"]


@pytest.fixture
def stdio_params(command: str, tools: Sequence[str]) -> MCPStdioParams:
    return MCPStdioParams(
        command=command,
        args=[],
        tools=tools,
    )


@pytest.fixture
def mcp_tools(tools: Sequence[str]) -> list[MCPTool]:
    return [
        MCPTool(name=tool, inputSchema={"type": "string", "properties": {}})
        for tool in tools
    ]


class ToolList(Protocol):
    @property
    def tools(self) -> list[MCPTool]: ...


@pytest.fixture
def tool_list(mcp_tools: Sequence[MCPTool]) -> ToolList:
    mock_tool_list = MagicMock()
    mock_tool_list.tools = mcp_tools
    return mock_tool_list


@pytest.fixture
def _patch_client_session_initialize() -> Generator[ClientSession]:
    with patch(
        "mcp.client.session.ClientSession.initialize",
        new_callable=AsyncMock,
        return_value=None,
    ):
        yield


@pytest.fixture
def _patch_client_session_list_tools(tool_list: ToolList) -> Generator[None]:
    with patch("mcp.client.session.ClientSession.list_tools") as mock_list_tools:
        mock_list_tools.return_value = tool_list
        yield


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


@pytest.fixture
def sse_params(echo_sse_server: Any, tools: Sequence[str]) -> MCPSseParams:
    return MCPSseParams(url=echo_sse_server["url"], tools=tools)


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    agent_framework: AgentFramework,
    sse_params: MCPSseParams,
    tools: Sequence[str],
) -> None:
    server = _get_mcp_server(sse_params, agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == set(tools)  # type: ignore[union-attr]
    else:
        assert len(server.tools) == len(tools)  # ignore[arg-type]
