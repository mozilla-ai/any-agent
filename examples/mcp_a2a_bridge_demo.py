"""Demo of MCP to A2A Bridge.

This example shows how to:
1. Expose MCP servers as A2A services
2. Use the bridged MCP tools from A2A agents
3. Integrate with mcpd's AGNTCY Identity
"""

import asyncio
import os
from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio
from any_agent.serving import (
    MCPToA2ABridgeConfig,
    serve_mcp_as_a2a_async,
)
from any_agent.tools import a2a_tool_async


async def demo_single_bridge():
    """Demo: Bridge a single MCP server to A2A."""
    print("=== Single MCP to A2A Bridge Demo ===\n")
    
    # Configure MCP server (using mcp-server-time as example)
    mcp_config = MCPStdio(
        command="uvx",
        args=["mcp-server-time", "--local-timezone=America/New_York"],
        tools=["get_current_time"],
    )
    
    # Create bridge configuration
    bridge_config = MCPToA2ABridgeConfig(
        mcp_config=mcp_config,
        port=9000,
        endpoint="/time-server",
        server_name="time-server",
        # If using mcpd with identity:
        # identity_id="did:agntcy:mcpd:demo:time-server"
    )
    
    # Start the bridge
    bridge_handle = await serve_mcp_as_a2a_async(mcp_config, bridge_config)
    
    try:
        print(f"✓ MCP server exposed as A2A service at: http://localhost:9000/time-server")
        
        # Create an agent that uses the bridged MCP tool via A2A
        print("\n--- Using from A2A agent ---")
        
        agent_config = AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Use the available tools to answer questions.",
            tools=[
                await a2a_tool_async(
                    "http://localhost:9000/time-server",
                    toolname="mcp_time_tool",
                    http_kwargs={"timeout": 30},
                ),
            ],
        )
        
        agent = await AnyAgent.create_async("tinyagent", agent_config)
        
        # Use the MCP tool through A2A
        result = await agent.run_async("What time is it in New York?")
        print(f"Agent response: {result.final_output}")
        
    finally:
        # Cleanup
        await bridge_handle.shutdown()


async def demo_multi_bridge():
    """Demo: Bridge multiple MCP servers."""
    print("\n=== Multiple MCP Bridges Demo ===\n")
    
    bridges = []
    tools = []
    
    try:
        # Bridge time server
        time_handle = await serve_mcp_as_a2a_async(
            MCPStdio(
                command="uvx",
                args=["mcp-server-time"],
                tools=["get_current_time"],
            ),
            MCPToA2ABridgeConfig(
                mcp_config=None,  # Will be set by serve function
                port=0,  # Dynamic port
                endpoint="/time",
                server_name="time",
            ),
        )
        bridges.append(time_handle)
        tools.append(
            await a2a_tool_async(
                f"http://localhost:{time_handle.port}/time",
                toolname="time_tool",
            )
        )
        
        print(f"✓ Bridged {len(bridges)} MCP servers to A2A")
        
        # Create orchestrator agent with all bridged tools
        orchestrator = await AnyAgent.create_async(
            "tinyagent",
            AgentConfig(
                model_id="mistral/mistral-small-latest",
                name="orchestrator",
                instructions="Use the available tools to answer questions.",
                tools=tools,
            ),
        )
        
        # Use the bridged tools
        result = await orchestrator.run_async("What time is it?")
        print(f"Orchestrator response: {result.final_output}")
        
    finally:
        # Cleanup all bridges
        for bridge in bridges:
            await bridge.shutdown()


async def main():
    """Run all demos."""
    # Set up API key
    if "MISTRAL_API_KEY" not in os.environ:
        print("Please set MISTRAL_API_KEY environment variable")
        return
    
    # Run demos
    await demo_single_bridge()
    await demo_multi_bridge()


if __name__ == "__main__":
    asyncio.run(main())