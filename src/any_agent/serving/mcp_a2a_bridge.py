"""MCP to A2A Bridge for any-agent.

This module provides a bridge that exposes MCP servers as A2A-compatible services,
enabling seamless tool-to-agent communication across protocols.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from any_agent.logging import logger
from any_agent.serving.server_handle import ServerHandle

if TYPE_CHECKING:
    from any_agent.config import MCPParams
    from any_agent.tools.mcp.mcp_client import MCPClient

# Check if A2A is available
a2a_available = False
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import TaskUpdater
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentSkill,
        Part,
        Role,
        TaskState,
        TextPart,
    )
    from a2a.utils import new_agent_parts_message
    
    a2a_available = True
except ImportError:
    pass


class MCPToA2ABridgeConfig(BaseModel):
    """Configuration for MCP to A2A bridge."""

    mcp_config: MCPParams
    host: str = Field(default="localhost", description="Host to serve on")
    port: int = Field(default=8080, description="Port to serve on")
    endpoint: str = Field(default="/mcp-bridge", description="Endpoint path")
    server_name: str = Field(default="mcp-server", description="MCP server name")
    identity_id: Optional[str] = Field(
        default=None, 
        description="AGNTCY Identity DID for internal tracking (not exposed via A2A)"
    )
    version: str = Field(default="1.0.0", description="Version of the bridge")


class MCPBridgeExecutor(AgentExecutor):
    """Executor that translates A2A requests to MCP tool calls."""

    def __init__(self, mcp_client: MCPClient, bridge_config: MCPToA2ABridgeConfig):
        """Initialize the executor.

        Args:
            mcp_client: Connected MCP client
            bridge_config: Bridge configuration
        """
        self.mcp_client = mcp_client
        self.bridge_config = bridge_config
        self._mcp_tools: dict[str, Any] = {}
        self._callable_tools: list[Any] = []
        self._tool_map: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize by loading MCP tools and caching callable tools."""
        raw_tools = await self.mcp_client.list_raw_tools()
        self._mcp_tools = {tool.name: tool for tool in raw_tools}
        
        # Cache callable tools for performance
        self._callable_tools = await self.mcp_client.list_tools()
        self._tool_map = {tool.__name__: tool for tool in self._callable_tools}
        
        logger.info(f"Loaded {len(self._mcp_tools)} MCP tools")
        logger.debug(f"Cached {len(self._tool_map)} callable tools: {list(self._tool_map.keys())}")
    

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute A2A request by calling appropriate MCP tool.

        Args:
            context: A2A request context
            event_queue: Event queue for updates
        """
        query = context.get_user_input()
        task = context.current_task
        
        if not task:
            logger.error("No task in context")
            return
            
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # For MCP bridge, we expect the query to specify the tool and args
            # Simple format: just pass the query as input to the first available tool
            if not self._mcp_tools:
                error_msg = "No MCP tools available"
                await updater.update_status(
                    TaskState.complete,
                    message=new_agent_parts_message(
                        [Part(root=TextPart(text=error_msg))],
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
                return
            
            # Use first tool if only one available, otherwise return error
            if len(self._mcp_tools) == 1:
                tool_name = list(self._mcp_tools.keys())[0]
                args = {"input": query}
            else:
                # Multiple tools, need more specific request
                error_msg = f"Multiple tools available: {list(self._mcp_tools.keys())}. Please specify which tool to use."
                await updater.update_status(
                    TaskState.complete,
                    message=new_agent_parts_message(
                        [Part(root=TextPart(text=error_msg))],
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
                return
            
            # Call MCP tool
            logger.info(f"Calling MCP tool '{tool_name}' with args: {args}")
            
            # Get the callable tool from cache (O(1) lookup)
            tool_func = self._tool_map.get(tool_name)
            
            if not tool_func:
                raise ValueError(f"Tool {tool_name} not found in callable tools")
            
            # Execute the tool
            result = await tool_func(**args) if asyncio.iscoroutinefunction(tool_func) else tool_func(**args)
            
            # Format result
            result_text = str(result)
            
            # Send success response
            await updater.update_status(
                TaskState.complete,
                message=new_agent_parts_message(
                    [Part(root=TextPart(text=result_text))],
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            
        except Exception as e:
            logger.error(f"Error executing MCP tool: {e}")
            await updater.update_status(
                TaskState.failed,
                message=new_agent_parts_message(
                    [Part(root=TextPart(text=f"Error: {str(e)}"))],
                    task.context_id,
                    task.id,
                ),
                final=True,
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported."""
        logger.warning("Cancel requested but not supported")


async def serve_mcp_as_a2a_async(
    mcp_config: MCPParams,
    bridge_config: Optional[MCPToA2ABridgeConfig] = None,
    framework: str = "tinyagent",
) -> ServerHandle:
    """Serve an MCP server as an A2A service.

    Args:
        mcp_config: MCP server configuration
        bridge_config: Bridge configuration (uses defaults if not provided)
        framework: Agent framework to use for tool conversion

    Returns:
        ServerHandle for managing the server

    Raises:
        ImportError: If A2A dependencies are not installed
    """
    if not a2a_available:
        msg = "You need to `pip install 'any-agent[a2a]'` to use MCP to A2A bridge"
        raise ImportError(msg)
    
    from any_agent.config import AgentFramework
    from any_agent.tools.mcp.mcp_client import MCPClient
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    
    # Use default config if not provided
    if bridge_config is None:
        bridge_config = MCPToA2ABridgeConfig(mcp_config=mcp_config)
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
    
    # Generate agent card
    mcp_tools = await mcp_client.list_raw_tools()
    skills = []
    for tool in mcp_tools:
        skill = AgentSkill(
            id=f"{bridge_config.server_name}-{tool.name}",
            name=tool.name,
            description=tool.description or f"MCP tool: {tool.name}",
            tags=[],
        )
        skills.append(skill)
    
    agent_card = AgentCard(
        name=f"{bridge_config.server_name}-bridge",
        description=f"MCP server '{bridge_config.server_name}' exposed via A2A",
        version=bridge_config.version,
        default_input_modes=["text"],
        default_output_modes=["text"],
        url=f"http://{bridge_config.host}:{bridge_config.port}{bridge_config.endpoint}",
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            state_transition_history=False,
        ),
        skills=skills,
    )
    
    # Note: Identity ID is stored in config but not exposed in AgentCard
    # as A2A's AgentCard may not support metadata field
    
    # Create A2A request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=None,  # Uses in-memory store
        push_config_store=None,
        push_sender=None,
    )
    
    # Create A2A app
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Mount under endpoint
    root = bridge_config.endpoint.lstrip("/").rstrip("/")
    app = Starlette(routes=[Mount(f"/{root}", routes=a2a_app.build().routes)])
    
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
    
    # Update agent card URL if using dynamic port
    if bridge_config.port == 0 and server.servers:
        actual_port = server.servers[0].sockets[0].getsockname()[1]
        agent_card.url = f"http://{bridge_config.host}:{actual_port}{bridge_config.endpoint}"
    
    if bridge_config.identity_id:
        logger.info(f"MCP to A2A bridge started at {agent_card.url} with identity {bridge_config.identity_id}")
    else:
        logger.info(f"MCP to A2A bridge started at {agent_card.url}")
    
    return ServerHandle(task=task, server=server)