# pylint: disable=unused-argument, unused-variable
import shutil
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import AsyncExitStack
from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.tools.mcp import MCPTools as AgnoMCPTools
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset as GoogleMCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import (  # type: ignore[attr-defined]
    SseServerParams as GoogleSseServerParameters,
)
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession

from any_agent.config import MCPSseParams, MCPStdioParams, Tool


class Toolset(Protocol):
    def load_tools(self) -> list[Tool]: ...


@pytest.fixture
def tools() -> list[Tool]:
    return ["write_file", "read_file", "other_tool"]


@pytest.fixture
def toolset(tools: Sequence[Tool]) -> Toolset:
    mock_value = AsyncMock()
    mock_value.load_tools.return_value = tools
    return mock_value


@pytest.fixture
def enter_context(
    toolset: Toolset,
) -> Generator[Callable[[], Toolset]]:
    with patch.object(AsyncExitStack, "enter_async_context") as mock_context:
        mock_context.return_value = toolset
        yield mock_context


@pytest.fixture
def mcp_sse_params_no_tools() -> MCPSseParams:
    return MCPSseParams(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
    )


@pytest.fixture
def mcp_sse_params_with_tools(
    mcp_sse_params_no_tools: MCPSseParams, tools: Sequence[Tool]
) -> MCPSseParams:
    return mcp_sse_params_no_tools.model_copy(update={"tools": tools})


# Google Specific fixtures


@pytest.fixture
def google_sse_params() -> Generator[GoogleSseServerParameters]:
    with patch(
        "any_agent.tools.mcp.frameworks.google.GoogleSseServerParameters"
    ) as mock_params:
        yield mock_params


@pytest.fixture
def google_toolset(toolset: Toolset) -> Generator[GoogleMCPToolset]:
    with patch("any_agent.tools.mcp.frameworks.google.GoogleMCPToolset") as mock_class:
        mock_class.return_value = toolset
        yield mock_class


# Agno Specific fixtures


@pytest.fixture
def session() -> Generator[Any]:
    return AsyncMock()


@pytest.fixture
def transport() -> tuple[Any, Any]:
    return (AsyncMock(), AsyncMock())


@pytest.fixture
def _path_client_session(session: AsyncGenerator[Any]) -> Generator[None]:
    with patch(
        "any_agent.tools.mcp.frameworks.agno.ClientSession"
    ) as mock_client_session:
        mock_client_session.return_value.__aenter__.return_value = session
        yield


@pytest.fixture
def agno_mcp_tool_instance() -> AgnoMCPTools:
    return MagicMock()


@pytest.fixture
def agno_mcp_tools(agno_mcp_tool_instance: AgnoMCPTools) -> Generator[AgnoMCPTools]:
    with patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools") as mock_mcp_tools:
        mock_mcp_tools.return_value = agno_mcp_tool_instance
        yield mock_mcp_tools


@pytest.fixture
def enter_context_with_transport_and_session(
    transport,
    session,
    tools: Sequence[str],
) -> Generator[None]:
    with patch.object(AsyncExitStack, "enter_async_context") as mock_context:
        mock_context.side_effect = [transport, session, tools]
        yield


# Specific for Stdio params


@pytest.fixture
def command() -> str:
    # Mocking the command part of stdio is really tricky so instead we'll use
    # a real command that should be available on all systems (this is what openai-agents does too)
    tee = shutil.which("tee") or ""
    assert tee, "tee not found"
    return tee


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


# Specific for echo server with sse params


@pytest.fixture
def sse_params_echo_server(echo_sse_server: Any, tools: Sequence[str]) -> MCPSseParams:
    return MCPSseParams(url=echo_sse_server["url"], tools=tools)
