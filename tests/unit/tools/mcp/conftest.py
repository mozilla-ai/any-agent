import shutil
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import AsyncExitStack
from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession
from pydantic import Field
from pytest_lazy_fixtures import lf

from any_agent.config import MCPParams, MCPSse, MCPStdio, Tool
from any_agent.tools import _MCPConnection


class Toolset(Protocol):
    def load_tools(self) -> list[Tool]: ...


@pytest.fixture
def tools() -> list[Tool]:
    return ["write_file", "read_file", "other_tool"]


@pytest.fixture
def mcp_sse_params_no_tools() -> MCPSse:
    return MCPSse(
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
    mcp_sse_params_no_tools: MCPSse, tools: Sequence[Tool]
) -> MCPSse:
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
def stdio_params(command: str, tools: Sequence[str]) -> MCPStdio:
    return MCPStdio(command=command, args=[], tools=tools)


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
def sse_params_echo_server(echo_sse_server: Any, tools: Sequence[str]) -> MCPSse:
    return MCPSse(url=echo_sse_server["url"], tools=tools)


class FakeMCPConnection(_MCPConnection[Any]):
    mcp_tool: None = None  # type: ignore[assignment]
    tools: Sequence[Tool] = Field(default_factory=list)

    async def list_tools(self) -> list[Tool]:
        return list(self.tools)


@pytest.fixture
def mcp_connection(tools: Sequence[Tool]) -> _MCPConnection[Any]:
    return FakeMCPConnection(tools=tools)


@pytest.fixture(
    params=[lf("stdio_params"), lf("mcp_sse_params_with_tools")], ids=["STDIO", "SSE"]
)
def mcp_params(request: pytest.FixtureRequest) -> MCPParams:
    return request.param  # type: ignore[no-any-return]
