# pylint: disable=unused-argument, unused-variable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.mark.asyncio
async def test_llamaindex_mcp_sse() -> None:
    """Test LlamaIndexMCPServer with SSE configuration."""
    # Mock the necessary components
    mock_tools = [MagicMock(), MagicMock()]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(url="http://localhost:8000/sse", tools=["tool1", "tool2"])

    # Create the server instance
    server = _get_mcp_server(mcp_tool, AgentFramework.LLAMA_INDEX)

    # Mock LlamaIndex MCP classes
    with (
        patch(
            "any_agent.tools.mcp.frameworks.llama_index.LlamaIndexMCPClient"
        ) as mock_client_class,
        patch(
            "any_agent.tools.mcp.frameworks.llama_index.LlamaIndexMcpToolSpec"
        ) as mock_tool_spec_class,
    ):
        # Set up mock client and tool spec
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_tool_spec = MagicMock()
        mock_tool_spec.to_tool_list_async = AsyncMock(return_value=mock_tools)
        mock_tool_spec_class.return_value = mock_tool_spec

        # Test the _setup_tools method
        await server._setup_tools()

        # Verify the client was created correctly
        mock_client_class.assert_called_once_with(
            command_or_url="http://localhost:8000/sse"
        )

        # Verify tool spec was created with correct params
        mock_tool_spec_class.assert_called_once_with(
            client=mock_client, allowed_tools=["tool1", "tool2"]
        )

        # Verify to_tool_list_async was called
        mock_tool_spec.to_tool_list_async.assert_called_once()

        # Check that tools were stored
        assert server.tools == mock_tools
