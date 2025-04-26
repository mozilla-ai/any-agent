from collections.abc import Sequence

import pytest

from any_agent.config import AgentConfig, AgentFramework, MCPSseParams, Tool
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import _get_mcp_server


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "enter_context_with_transport_and_session", "openai_mcp_sse_server"
)
async def test_openai_mcp_sse_tools_loaded(
    tools: Sequence[Tool],
    mcp_sse_params_no_tools: MCPSseParams,
) -> None:
    mcp_server = _get_mcp_server(mcp_sse_params_no_tools, AgentFramework.OPENAI)
    await mcp_server._setup_tools()

    assert mcp_server.tools == tools


@pytest.mark.usefixtures(
    "openai_mcp_sse_server",
)
def test_openai_mcpsse(
    mcp_sse_params_no_tools: MCPSseParams,
) -> None:
    """This is a test kept for legacy purposes."""
    agent_config = AgentConfig(model_id="gpt-4o", tools=[mcp_sse_params_no_tools])

    agent = AnyAgent.create("openai", agent_config)

    servers = agent._mcp_servers
    assert servers

    server, *_ = agent._mcp_servers
    assert server.mcp_tool == mcp_sse_params_no_tools
