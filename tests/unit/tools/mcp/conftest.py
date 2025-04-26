import shutil
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import AsyncExitStack
from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession

from any_agent.config import MCPSseParams, MCPStdioParams, Tool


class Toolset(Protocol):
    def load_tools(self) -> list[Tool]: ...


@pytest.fixture
def tools() -> list[Tool]:
    return ["write_file", "read_file", "other_tool"]


@pytest.fixture
def mcp_sse_params_no_tools() -> MCPSseParams:
    return MCPSseParams(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
    )


@pytest.fixture
def session() -> Generator[Any]:
    return AsyncMock()


@pytest.fixture
def _path_client_session(session: AsyncGenerator[Any]) -> Generator[None]:
    with patch(
        "any_agent.tools.mcp.frameworks.agno.ClientSession"
    ) as mock_client_session:
        mock_client_session.return_value.__aenter__.return_value = session
        yield


@pytest.fixture
def mcp_sse_params_with_tools(
    mcp_sse_params_no_tools: MCPSseParams, tools: Sequence[Tool]
) -> MCPSseParams:
    return mcp_sse_params_no_tools.model_copy(update={"tools": tools})


@pytest.fixture
def enter_context_with_transport_and_session(
    session: Any,
    tools: Sequence[str],
) -> Generator[None]:
    transport = (AsyncMock(), AsyncMock())
    with patch.object(AsyncExitStack, "enter_async_context") as mock_context:
        mock_context.side_effect = [transport, session, tools]
        yield


@pytest.fixture
def command() -> str:
    # Mocking the command part of stdio is really tricky so instead we'll use
    # a real command that should be available on all systems (this is what openai-agents does too)
    tee = shutil.which("tee") or ""
    assert tee, "tee not found"
    return tee


@pytest.fixture
def stdio_params(command: str, tools: Sequence[str]) -> MCPStdioParams:
    return MCPStdioParams(command=command, args=[], tools=tools)


@pytest.fixture
def mcp_tools(tools: Sequence[str]) -> list[MCPTool]:
    return [
        MCPTool(name=tool, inputSchema={"type": "string", "properties": {}})
        for tool in tools
    ]


@pytest.fixture
def _patch_client_session_initialize() -> Generator[ClientSession]:
    with patch(
        "mcp.client.session.ClientSession.initialize",
        new_callable=AsyncMock,
        return_value=None,
    ) as initialize_mock:
        yield initialize_mock


@pytest.fixture
def _patch_client_session_list_tools(mcp_tools: Sequence[MCPTool]) -> Generator[None]:
    tool_list = MagicMock()
    tool_list.tools = mcp_tools
    with patch("mcp.client.session.ClientSession.list_tools", return_value=tool_list):
        yield


@pytest.fixture
def sse_params_echo_server(echo_sse_server: Any, tools: Sequence[str]) -> MCPSseParams:
    return MCPSseParams(url=echo_sse_server["url"], tools=tools)
