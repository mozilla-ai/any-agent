# pylint: disable=unused-argument, unused-variable
from unittest.mock import AsyncMock, MagicMock, patch

from any_agent.config import AgentConfig, AgentFramework, MCPSseParams
from any_agent.frameworks.any_agent import AnyAgent


def test_openai_mcpsse() -> None:
    agent_framework = AgentFramework.OPENAI
    # Set up our mocks
    mock_server = AsyncMock()
    from mcp import Tool as MCPTool

    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_server._tools_list = [mock_tool]  # pylint: disable=protected-access

    # Path the imports and class
    with patch(
        "any_agent.tools.mcp.frameworks.openai.OpenAIInternalMCPServerSse",
        return_value=mock_server,
    ):
        # Set up tools config for agent
        tools = [MCPSseParams(url="http://localhost:8000/sse")]

        # Create and run the agent
        agent_config = AgentConfig(model_id="gpt-4o", tools=tools)

        agent = AnyAgent.create(agent_framework, agent_config)
        assert len(agent._mcp_servers) > 0
