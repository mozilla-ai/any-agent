# pylint: disable=unused-argument, unused-variable
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentFramework, MCPSseParams
from any_agent.tools.mcp.frameworks import _get_mcp_server


@pytest.mark.asyncio
async def test_google_mcp_sse() -> None:
    """Test GoogleMCPServer with SSE configuration."""
    # Mock the necessary components
    mock_tools = [MagicMock(), MagicMock()]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test-token"},
    )

    # Create the server instance
    server = _get_mcp_server(mcp_tool, AgentFramework.GOOGLE)

    # Mock Google MCP classes
    with (
        patch(
            "any_agent.tools.mcp.frameworks.google.GoogleMCPToolset"
        ) as mock_toolset_class,
        patch(
            "any_agent.tools.mcp.frameworks.google.GoogleSseServerParameters"
        ) as mock_sse_params,
    ):
        # Set up mock toolset
        mock_toolset = AsyncMock()
        mock_toolset.load_tools.return_value = mock_tools
        mock_toolset_class.return_value = mock_toolset

        # Mock AsyncExitStack to avoid actually setting up exit handlers
        with patch.object(AsyncExitStack, "enter_async_context") as mock_enter_context:
            mock_enter_context.return_value = mock_toolset

            # Test the _setup_tools method
            await server._setup_tools()

            # Verify the SseServerParams was created correctly
            mock_sse_params.assert_called_once_with(
                url="http://localhost:8000/sse",
                headers={"Authorization": "Bearer test-token"},
            )

            # Verify toolset was created with correct params
            mock_toolset_class.assert_called_once_with(
                connection_params=mock_sse_params.return_value
            )

            # Verify tools were loaded
            mock_toolset.load_tools.assert_called_once()

            # Check that tools were stored
            assert server.tools == mock_tools
            assert server.server == mock_toolset  # type: ignore[union-attr]
