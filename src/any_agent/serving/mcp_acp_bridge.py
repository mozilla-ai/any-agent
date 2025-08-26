"""MCP to ACP Bridge for any-agent.

This module provides a bridge that exposes MCP servers as ACP-compatible services,
enabling seamless tool-to-agent communication across protocols.

Following the pattern established in mcp_a2a_bridge.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from any_agent.logging import logger
from any_agent.serving.server_handle import ServerHandle

if TYPE_CHECKING:
    from any_agent.config import MCPParams
    from any_agent.tools.mcp.mcp_client import MCPClient

# Check if ACP is available
acp_available = False
try:
    from agntcy_acp import (
        ACPAsyncClient,
        Agent,
        AgentACPDescriptor,
        AgentACPSpec,
        AgentCapabilities,
        AgentMetadata,
        RunCreateStateless,
        RunStateless,
        RunStatus,
        StreamingMode,
        StreamingModes,
    )
    
    acp_available = True
except ImportError:
    pass


class MCPToACPBridgeConfig(BaseModel):
    """Configuration for MCP to ACP bridge."""

    mcp_config: MCPParams
    host: str = Field(default="localhost", description="Host to serve on")
    port: int = Field(default=8090, description="Port to serve on")
    endpoint: str = Field(default="/mcp-bridge", description="Endpoint path")
    server_name: str = Field(default="mcp-server", description="MCP server name")
    identity_id: Optional[str] = Field(
        default=None, 
        description="AGNTCY Identity DID for the bridge"
    )
    version: str = Field(default="1.0.0", description="Version of the bridge")
    organization: str = Field(default="any-agent", description="Organization name")


class MCPBridgeExecutor:
    """Executor that translates ACP requests to MCP tool calls."""

    def __init__(self, mcp_client: MCPClient, bridge_config: MCPToACPBridgeConfig):
        """Initialize the executor.

        Args:
            mcp_client: Connected MCP client
            bridge_config: Bridge configuration
        """
        self.mcp_client = mcp_client
        self.bridge_config = bridge_config
        self._mcp_tools: dict[str, Any] = {}
        self._agent_manifest: Optional[dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize by loading MCP tools and creating ACP manifest."""
        raw_tools = await self.mcp_client.list_raw_tools()
        self._mcp_tools = {tool.name: tool for tool in raw_tools}
        logger.info(f"Loaded {len(self._mcp_tools)} MCP tools")
        
        # Create ACP manifest
        self._agent_manifest = await self._create_acp_manifest()
    
    async def _create_acp_manifest(self) -> dict[str, Any]:
        """Create ACP manifest from MCP tools."""
        mcp_tools = await self.mcp_client.list_raw_tools()
        
        # Build tool descriptions for manifest
        tools_description = []
        for tool in mcp_tools:
            tools_description.append({
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            })
        
        manifest = {
            "id": f"mcp-bridge-{self.bridge_config.server_name}",
            "name": f"{self.bridge_config.server_name} MCP Bridge",
            "version": self.bridge_config.version,
            "description": f"MCP server '{self.bridge_config.server_name}' exposed via ACP",
            "metadata": {
                "organization": self.bridge_config.organization,
                "bridge_type": "mcp-to-acp",
                "identity_id": self.bridge_config.identity_id,
            },
            "acp": {
                "version": "0.2.3",
                "capabilities": {
                    "stateless": True,
                    "stateful": False,
                    "threads": False,
                    "callbacks": False,
                    "streaming": StreamingModes(
                        stateless=StreamingMode.VALUE,
                        stateful=None
                    ) if acp_available else {"stateless": "value", "stateful": None},
                    "interrupts": False,
                },
                "input": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "Tool name to call"},
                        "args": {"type": "object", "description": "Tool arguments"}
                    },
                    "required": ["tool"]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "result": {"description": "Tool execution result"},
                        "error": {"type": "string", "description": "Error message if failed"}
                    }
                },
                "tools": tools_description,
            }
        }
        
        return manifest

    async def execute_stateless_run(self, run_request: RunCreateStateless) -> RunStateless:
        """Execute a stateless ACP run by calling appropriate MCP tool.

        Args:
            run_request: ACP run request

        Returns:
            ACP run result
        """
        run_id = str(uuid4())
        
        try:
            # Extract tool and args from config
            config = run_request.config or {}
            tool_name = config.get("tool")
            args = config.get("args", {})
            
            if not tool_name:
                # If no tool specified, try to infer from input
                # This is a simple heuristic - could be improved
                if len(self._mcp_tools) == 1:
                    tool_name = list(self._mcp_tools.keys())[0]
                    args = {"input": run_request.input} if run_request.input else {}
                else:
                    raise ValueError(
                        f"Multiple tools available: {list(self._mcp_tools.keys())}. "
                        "Please specify which tool to use in config.tool"
                    )
            
            # Validate tool exists
            if tool_name not in self._mcp_tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Call MCP tool
            logger.info(f"Calling MCP tool '{tool_name}' with args: {args}")
            
            # Get the callable tool
            tools = await self.mcp_client.list_tools()
            tool_func = next((t for t in tools if t.__name__ == tool_name), None)
            
            if not tool_func:
                raise ValueError(f"Tool {tool_name} not found in callable tools")
            
            # Execute the tool
            result = await tool_func(**args) if asyncio.iscoroutinefunction(tool_func) else tool_func(**args)
            
            # Create successful run result
            return RunStateless(
                id=run_id,
                status=RunStatus.completed,
                output={
                    "result": result,
                    "tool": tool_name,
                    "success": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing MCP tool: {e}")
            return RunStateless(
                id=run_id,
                status=RunStatus.failed,
                error={
                    "type": "ToolExecutionError",
                    "message": str(e)
                },
                output={
                    "error": str(e),
                    "success": False
                }
            )


async def serve_mcp_as_acp_async(
    mcp_config: MCPParams,
    bridge_config: Optional[MCPToACPBridgeConfig] = None,
    framework: str = "tinyagent",
) -> ServerHandle:
    """Serve an MCP server as an ACP service.

    Args:
        mcp_config: MCP server configuration
        bridge_config: Bridge configuration (uses defaults if not provided)
        framework: Agent framework to use for tool conversion

    Returns:
        ServerHandle for managing the server

    Raises:
        ImportError: If ACP dependencies are not installed
    """
    if not acp_available:
        msg = "You need to `pip install 'agntcy-acp'` to use MCP to ACP bridge"
        raise ImportError(msg)
    
    from any_agent.config import AgentFramework
    from any_agent.tools.mcp.mcp_client import MCPClient
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    # Use default config if not provided
    if bridge_config is None:
        bridge_config = MCPToACPBridgeConfig(mcp_config=mcp_config)
    else:
        # Ensure mcp_config is set
        bridge_config.mcp_config = mcp_config
    
    # Create and connect MCP client
    mcp_client = MCPClient(
        config=mcp_config,
        framework=AgentFramework.from_string(framework),
    )
    await mcp_client.connect()
    
    # Create executor
    executor = MCPBridgeExecutor(mcp_client, bridge_config)
    await executor.initialize()
    
    # ACP route handlers
    async def get_agents(request):
        """List available agents (in this case, just our bridge)."""
        agent = Agent(
            id=f"mcp-bridge-{bridge_config.server_name}",
            name=f"{bridge_config.server_name} MCP Bridge",
            description=f"MCP server '{bridge_config.server_name}' exposed via ACP",
            metadata=AgentMetadata(
                organization=bridge_config.organization,
                version=bridge_config.version,
            ),
            acp_descriptor=executor._agent_manifest.get("acp", {}),
        )
        return JSONResponse([agent.model_dump()])
    
    async def search_agents(request):
        """Search agents - returns our bridge if it matches."""
        # For simplicity, always return our agent
        return await get_agents(request)
    
    async def get_agent_by_id(request):
        """Get specific agent by ID."""
        agent_id = request.path_params["agent_id"]
        expected_id = f"mcp-bridge-{bridge_config.server_name}"
        
        if agent_id != expected_id:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        
        agent = Agent(
            id=expected_id,
            name=f"{bridge_config.server_name} MCP Bridge",
            description=f"MCP server '{bridge_config.server_name}' exposed via ACP",
            metadata=AgentMetadata(
                organization=bridge_config.organization,
                version=bridge_config.version,
            ),
            acp_descriptor=executor._agent_manifest.get("acp", {}),
        )
        return JSONResponse(agent.model_dump())
    
    async def create_stateless_run(request):
        """Create a stateless run."""
        body = await request.json()
        run_request = RunCreateStateless(**body)
        
        # Execute the run
        result = await executor.execute_stateless_run(run_request)
        
        return JSONResponse(result.model_dump())
    
    async def get_stateless_run(request):
        """Get stateless run status - not implemented for bridge."""
        return JSONResponse(
            {"error": "Run history not supported in bridge mode"}, 
            status_code=501
        )
    
    # Create Starlette app with ACP routes
    base_path = bridge_config.endpoint.rstrip("/")
    routes = [
        Route(f"{base_path}/agents/search", search_agents, methods=["POST"]),
        Route(f"{base_path}/agents", get_agents, methods=["GET"]),
        Route(f"{base_path}/agents/{{agent_id}}", get_agent_by_id, methods=["GET"]),
        Route(f"{base_path}/runs/stateless", create_stateless_run, methods=["POST"]),
        Route(f"{base_path}/runs/stateless/{{run_id}}", get_stateless_run, methods=["GET"]),
    ]
    
    app = Starlette(routes=routes)
    
    # Create and start server
    config = uvicorn.Config(
        app,
        host=bridge_config.host,
        port=bridge_config.port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    
    # Start server in background
    task = asyncio.create_task(server.serve())
    
    # Wait for server to start
    while not server.started:
        await asyncio.sleep(0.1)
    
    # Get actual port if using dynamic port
    actual_port = bridge_config.port
    if bridge_config.port == 0 and server.servers:
        actual_port = server.servers[0].sockets[0].getsockname()[1]
    
    bridge_url = f"http://{bridge_config.host}:{actual_port}{bridge_config.endpoint}"
    
    if bridge_config.identity_id:
        logger.info(f"MCP to ACP bridge started at {bridge_url} with identity {bridge_config.identity_id}")
    else:
        logger.info(f"MCP to ACP bridge started at {bridge_url}")
    
    logger.info(f"ACP manifest available at: {bridge_url}/agents")
    
    return ServerHandle(task=task, server=server)