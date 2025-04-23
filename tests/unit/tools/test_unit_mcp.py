# pylint: disable=unused-argument, unused-variable
# Test MCP Tools Classes.
# Disclaim

import asyncio
import unittest
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent.config import AgentConfig, AgentFramework, MCPSseParams, MCPStdioParams
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools import get_mcp_server


# Common helper functions for all test classes
def create_mock_tools() -> list[MagicMock]:
    """Helper method to create mock tools."""
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    return [mock_tool1, mock_tool2]


def create_specific_mock_tools() -> list[MagicMock]:
    """Helper method to create specific mock tools."""
    mock_read_tool = MagicMock()
    mock_read_tool.name = "read_thing"
    mock_write_tool = MagicMock()
    mock_write_tool.name = "write_thing"
    mock_other_tool = MagicMock()
    mock_other_tool.name = "other_thing"
    return [mock_read_tool, mock_write_tool, mock_other_tool]


@patch("any_agent.tools.mcp.frameworks.smolagents.MCPClient")
class TestSmolagentsMCPServer(unittest.TestCase):
    """Tests for the SmolagentsMCPServer class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test."""
        # Common test data
        self.test_tool = MagicMock(spec=MCPStdioParams)
        self.test_tool.command = "test_command"
        self.test_tool.args = ["arg1", "arg2"]

    def test_setup_tools_with_none_tools(
        self,
        mock_client_class: Any,
    ) -> None:
        """Test that when mcp_tool.tools is None, all available tools are used."""
        # Setup mock tools
        mock_tools = create_mock_tools()

        # Setup mock MCPClient context manager behavior
        mock_client_class.return_value.__enter__.return_value = mock_tools

        self.test_tool.tools = None
        mcp_server = get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify all tools are included
        assert mcp_server.tools == mock_tools
        assert len(mcp_server.tools) == 2

    def test_setup_tools_with_specific_tools(
        self,
        mock_client_class: Any,
    ) -> None:
        """Test that when mcp_tool.tools has specific values, only those tools are used."""
        # Setup mock tools
        mock_tools = create_specific_mock_tools()

        # Setup mock MCPClient context manager behavior
        mock_client_class.return_value.__enter__.return_value = mock_tools

        # Create test tool configuration with specific tools
        self.test_tool.tools = ["read_thing", "write_thing"]

        mcp_server = get_mcp_server(self.test_tool, AgentFramework.SMOLAGENTS)
        asyncio.get_event_loop().run_until_complete(mcp_server.setup_tools())

        # Verify only the requested tools are included
        assert len(mcp_server.tools) == 2
        tool_names = [tool.name for tool in mcp_server.tools]  # type: ignore[union-attr]
        assert "read_thing" in tool_names
        assert "write_thing" in tool_names
        assert "other_thing" not in tool_names


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


@pytest.mark.asyncio
async def test_smolagents_mcp_sse() -> None:
    # Create mock tools
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool1"
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool2"
    mock_tools = [mock_tool1, mock_tool2]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(url="http://localhost:8000/sse")

    # Create the server instance
    server = get_mcp_server(mcp_tool, AgentFramework.SMOLAGENTS)

    # Patch the MCPClient class to return our mock tools
    with patch(
        "any_agent.tools.mcp.frameworks.smolagents.MCPClient"
    ) as mock_client_class:
        # Setup the mock to return our tools when used as a context manager
        mock_client_class.return_value.__enter__.return_value = mock_tools

        # Test the setup_tools method
        await server.setup_tools()

        # Verify the client was created with correct parameters
        mock_client_class.assert_called_once_with({"url": "http://localhost:8000/sse"})

        # Verify tools were correctly assigned
        assert server.tools == mock_tools


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
        server = get_mcp_server(mcp_tool, AgentFramework.LANGCHAIN)

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

            # Test the setup_tools method
            await server.setup_tools()
            # Verify session was initialized
            mock_session.initialize.assert_called_once()
            # Verify tools were loaded
            mock_load_tools.assert_called_once_with(mock_session)
            # Check that tools were stored
            assert server.tools == mock_tools


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
    server = get_mcp_server(mcp_tool, AgentFramework.GOOGLE)

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

            # Test the setup_tools method
            await server.setup_tools()

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


@pytest.mark.asyncio
async def test_llamaindex_mcp_sse() -> None:
    """Test LlamaIndexMCPServer with SSE configuration."""
    # Mock the necessary components
    mock_tools = [MagicMock(), MagicMock()]

    # Create an MCP tool config for SSE
    mcp_tool = MCPSseParams(url="http://localhost:8000/sse", tools=["tool1", "tool2"])

    # Create the server instance
    server = get_mcp_server(mcp_tool, AgentFramework.LLAMA_INDEX)

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

        # Test the setup_tools method
        await server.setup_tools()

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
    server = get_mcp_server(mcp_tool, AgentFramework.AGNO)

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

            # Test the setup_tools method
            await server.setup_tools()

            # Verify session was initialized
            mock_session.initialize.assert_called_once()

            # Verify MCPTools was created with correct params
            mock_mcp_tools.assert_called_once_with(
                session=mock_session, include_tools=["tool1", "tool2"]
            )

            # Check that tools instance was set as server
            assert server.server == mock_tools_instance  # type: ignore[union-attr]
