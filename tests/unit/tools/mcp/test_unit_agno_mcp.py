# pylint: disable=unused-argument, unused-variable
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.mark.asyncio
async def test_agno_mcp_sse() -> None:
    """Test AgnoMCPToolConnection with SSE configuration."""
    # Mock the necessary components
    mock_tools = [MagicMock(), MagicMock()]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
        tools=["tool1", "tool2"],
    )

    # Create the server instance
    server = _get_mcp_server(mcp_tool, AgentFramework.AGNO)

    # Mock required components
    with (
        patch(
            "any_agent.tools.mcp.frameworks.agno.ClientSession"
        ) as mock_client_session,
        patch("any_agent.tools.mcp.frameworks.agno.AgnoMCPTools") as mock_mcp_tools,
    ):
        # Set up mocks
        mock_transport = (AsyncMock(), AsyncMock())

        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        mock_tools_instance = MagicMock()
        mock_mcp_tools.return_value = mock_tools_instance

        # Mock AsyncExitStack to avoid actually setting up exit handlers
        with patch.object(AsyncExitStack, "enter_async_context") as mock_enter_context:
            mock_enter_context.side_effect = [mock_transport, mock_session, mock_tools]

            # Test the _setup_tools method
            await server._setup_tools()

            # Verify session was initialized
            mock_session.initialize.assert_called_once()

            # Verify MCPTools was created with correct params
            mock_mcp_tools.assert_called_once_with(
                session=mock_session, include_tools=["tool1", "tool2"]
            )

            # Check that tools instance was set as server
            assert server.server == mock_tools_instance  # type: ignore[union-attr]
