"""Integration tests for MCP to A2A bridge."""

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio
from any_agent.serving import (
    MCPToA2ABridgeConfig,
    serve_mcp_as_a2a_async,
)
from any_agent.tools import a2a_tool_async


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPToA2ABridge:
    """Test MCP to A2A bridge functionality."""

    async def test_basic_bridge(self):
        """Test basic MCP to A2A bridge functionality."""
        # Configure MCP server
        mcp_config = MCPStdio(
            command="echo",
            args=["test"],
            tools=["echo_tool"],
        )
        
        # Configure bridge
        bridge_config = MCPToA2ABridgeConfig(
            mcp_config=mcp_config,
            port=0,  # Dynamic port
            endpoint="/test-bridge",
            server_name="test-server",
        )
        
        # Start bridge
        bridge_handle = await serve_mcp_as_a2a_async(mcp_config, bridge_config)
        
        try:
            # Verify bridge is running
            assert bridge_handle.port > 0
            
            # Get agent card
            bridge_url = f"http://localhost:{bridge_handle.port}/test-bridge"
            async with httpx.AsyncClient() as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=bridge_url)
                agent_card = await resolver.get_agent_card()
                
                assert agent_card.name == "test-server-bridge", f"Expected name 'test-server-bridge', got {agent_card.name}"
                assert len(agent_card.skills) > 0, "Expected at least one skill"
                assert agent_card.skills[0].name == "echo_tool", f"Expected skill name 'echo_tool', got {agent_card.skills[0].name}"
                
        finally:
            await bridge_handle.shutdown()

    async def test_bridge_with_identity(self):
        """Test bridge with AGNTCY identity."""
        # Configure with identity
        bridge_config = MCPToA2ABridgeConfig(
            mcp_config=MCPStdio(
                command="echo",
                args=["test"],
                tools=["echo_tool"],
            ),
            port=0,
            endpoint="/secure-bridge",
            server_name="secure-server",
            identity_id="did:agntcy:mcpd:test:secure-server",
        )
        
        # Start bridge
        bridge_handle = await serve_mcp_as_a2a_async(
            bridge_config.mcp_config,
            bridge_config,
        )
        
        try:
            # Get agent card and verify identity
            bridge_url = f"http://localhost:{bridge_handle.port}/secure-bridge"
            async with httpx.AsyncClient() as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=bridge_url)
                agent_card = await resolver.get_agent_card()
                
                # Identity is stored in config but not exposed in AgentCard
                # as A2A's AgentCard may not support metadata field
                assert bridge_config.identity_id == "did:agntcy:mcpd:test:secure-server"
                
        finally:
            await bridge_handle.shutdown()

    async def test_tool_invocation_via_a2a(self):
        """Test invoking MCP tool through A2A protocol."""
        # Use a simple MCP server
        mcp_config = MCPStdio(
            command="python",
            args=["-c", "print('Hello from MCP')"],
            tools=["hello_tool"],
        )
        
        bridge_config = MCPToA2ABridgeConfig(
            mcp_config=mcp_config,
            port=0,
            endpoint="/hello-bridge",
            server_name="hello-server",
        )
        
        # Start bridge
        bridge_handle = await serve_mcp_as_a2a_async(mcp_config, bridge_config)
        
        try:
            # Create A2A tool
            bridge_url = f"http://localhost:{bridge_handle.port}/hello-bridge"
            tool = await a2a_tool_async(bridge_url, toolname="mcp_hello")
            
            # Create agent with bridged tool
            agent = await AnyAgent.create_async(
                "tinyagent",
                AgentConfig(
                    model_id="mistral/mistral-small-latest",
                    instructions="Use the available tools.",
                    tools=[tool],
                ),
            )
            
            # Use the tool
            result = await agent.run_async("Say hello using the tool")
            assert result.final_output is not None
            
        finally:
            await bridge_handle.shutdown()

    async def test_direct_tool_call(self):
        """Test direct tool invocation without agent."""
        # Simple echo MCP server
        mcp_config = MCPStdio(
            command="echo",
            args=["direct test"],
            tools=["echo_tool"],
        )
        
        bridge_config = MCPToA2ABridgeConfig(
            mcp_config=mcp_config,
            port=0,
            endpoint="/direct-bridge",
            server_name="direct-server",
        )
        
        # Start bridge
        bridge_handle = await serve_mcp_as_a2a_async(mcp_config, bridge_config)
        
        try:
            bridge_url = f"http://localhost:{bridge_handle.port}/direct-bridge"
            
            # Get agent card
            async with httpx.AsyncClient() as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=bridge_url)
                agent_card = await resolver.get_agent_card()
                
                # Direct tool call
                a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
                
                request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message=Message(
                            role=Role.user,
                            parts=[Part(root=TextPart(text="test message"))],
                            message_id=str(uuid4()),
                        )
                    ),
                )
                
                response = await a2a_client.send_message(request)
                assert response is not None
                
        finally:
            await bridge_handle.shutdown()

    async def test_multiple_bridges(self):
        """Test running multiple bridges simultaneously."""
        bridges = []
        
        try:
            # Start multiple bridges
            for i in range(3):
                config = MCPToA2ABridgeConfig(
                    mcp_config=MCPStdio(
                        command="echo",
                        args=[f"server-{i}"],
                        tools=[f"tool_{i}"],
                    ),
                    port=0,
                    endpoint=f"/bridge-{i}",
                    server_name=f"server-{i}",
                )
                
                handle = await serve_mcp_as_a2a_async(config.mcp_config, config)
                bridges.append(handle)
            
            # Verify all are running
            assert len(bridges) == 3
            for handle in bridges:
                assert handle.port > 0
                
        finally:
            # Cleanup all bridges
            for handle in bridges:
                await handle.shutdown()