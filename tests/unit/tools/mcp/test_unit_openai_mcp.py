import pytest

from any_agent.config import AgentConfig, MCPSseParams
from any_agent.frameworks.any_agent import AnyAgent


@pytest.mark.usefixtures(
    "openai_mcp_sse_server",
)
def test_openai_mcpsse(
    mcp_sse_params_no_tools: MCPSseParams,
) -> None:
    agent_config = AgentConfig(model_id="gpt-4o", tools=[mcp_sse_params_no_tools])

    agent = AnyAgent.create("openai", agent_config)
    
    servers = agent._mcp_servers
    assert servers

    server, *_ = agent._mcp_servers
    assert server.mcp_tool == mcp_sse_params_no_tools
