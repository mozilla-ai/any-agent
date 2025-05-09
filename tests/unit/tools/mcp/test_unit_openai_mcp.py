from collections.abc import Generator, Sequence
from unittest.mock import AsyncMock, patch

import pytest
from agents.mcp import MCPServerSse as OpenAIInternalMCPServerSse

from any_agent.config import AgentConfig, MCPSse, Tool
from any_agent.frameworks.any_agent import AnyAgent


@pytest.fixture
def openai_mcp_sse_server(
    tools: Sequence[Tool],
) -> Generator[OpenAIInternalMCPServerSse]:
    with patch(
        "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerSse",
    ) as mock_server:
        mock_server.return_value.list_tools = AsyncMock(return_value=tools)
        yield mock_server


@pytest.mark.usefixtures(
    "openai_mcp_sse_server",
)
def test_openai_mcpsse(
    mcp_sse_params_no_tools: MCPSse,
) -> None:
    """This is a test kept for legacy purposes."""
    agent_config = AgentConfig(model_id="gpt-4o", tools=[mcp_sse_params_no_tools])

    agent = AnyAgent.create("openai", agent_config)

    servers = agent._mcp_servers
    assert servers

    server, *_ = agent._mcp_servers
    assert server.mcp_tool == mcp_sse_params_no_tools
