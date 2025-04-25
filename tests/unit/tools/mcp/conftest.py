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

from any_agent.config import MCPSseParams, Tool


class Toolset(Protocol):
    def load_tools(self) -> list[Tool]: ...


@pytest.fixture
def tools() -> list[Tool]:
    return [MagicMock(), MagicMock()]


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
