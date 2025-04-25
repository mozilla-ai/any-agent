# pylint: disable=unused-argument, unused-variable
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.mark.asyncio
async def test_langchain_mcp_sse() -> None:
    """Test LangchainMCPServer with SSE configuration."""
    # Mock the necessary components
    mock_tools = [MagicMock(), MagicMock()]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
    )

    # Mock required components
    with (
        patch(
            "any_agent.tools.mcp.frameworks.langchain.load_mcp_tools"
        ) as mock_load_tools,
        patch("mcp.ClientSession") as mock_client_session,
    ):
        # Create the server instance
        server = _get_mcp_server(mcp_tool, AgentFramework.LANGCHAIN)

        # Set up mocks
        mock_transport = (AsyncMock(), AsyncMock())

        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        mock_load_tools.return_value = mock_tools

        # Mock AsyncExitStack to avoid actually setting up exit handlers
        with patch.object(AsyncExitStack, "enter_async_context") as mock_enter_context:
            mock_enter_context.side_effect = [
                mock_transport,
                mock_session,
            ]

            # Test the _setup_tools method
            await server._setup_tools()
            # Verify session was initialized
            mock_session.initialize.assert_called_once()
            # Verify tools were loaded
            mock_load_tools.assert_called_once_with(mock_session)
            # Check that tools were stored
            assert server.tools == mock_tools
