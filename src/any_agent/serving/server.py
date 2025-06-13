from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import mcp.types as mcptypes
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from mcp.server import Server as MCPServer
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from any_agent.serving.task_manager import TaskManager
from any_agent.utils import run_async_in_sync

from .agent_card import _build_agent_card
from .agent_executor import AnyAgentExecutor
from .envelope import prepare_agent_for_a2a, prepare_agent_for_a2a_async

if TYPE_CHECKING:
    from multiprocessing import Queue
    from starlette.requests import Request

    from any_agent import AnyAgent
    from any_agent.serving import A2AServingConfig


def _get_a2a_app(
    agent: AnyAgent, serving_config: A2AServingConfig
) -> A2AStarletteApplication:
    agent = prepare_agent_for_a2a(agent)

    agent_card = _build_agent_card(agent, serving_config)
    task_manager = TaskManager(serving_config)

    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent, task_manager),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)


async def _get_a2a_app_async(
    agent: AnyAgent, serving_config: A2AServingConfig
) -> A2AStarletteApplication:
    agent = await prepare_agent_for_a2a_async(agent)

    agent_card = _build_agent_card(agent, serving_config)
    task_manager = TaskManager(serving_config)

    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent, task_manager),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)


def _create_server(
    app: A2AStarletteApplication,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> uvicorn.Server:
    root = endpoint.lstrip("/").rstrip("/")
    a2a_app = app.build()
    internal_router = Starlette(routes=[Mount(f"/{root}", routes=a2a_app.routes)])

    config = uvicorn.Config(internal_router, host=host, port=port, log_level=log_level)
    return uvicorn.Server(config)


def _create_mcp_server_instance(agent: AnyAgent) -> MCPServer:
    server = MCPServer("any-agent-mcp-server")

    @server.list_tools()
    async def handle_list_tools() -> list[mcptypes.Tool]:
        return [
            mcptypes.Tool(
                name=f"as-tool-{agent.config.name}",
                description=agent.config.description,
                inputSchema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The prompt for the agent",
                        },
                    },
                },
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[mcptypes.TextContent | mcptypes.ImageContent | mcptypes.EmbeddedResource]:
        result = await agent.run_async(arguments["query"])
        return [mcptypes.TextContent(type="text", text=result.final_output)]

    @server.list_resource_templates()
    async def handle_list_resource_templates() -> list[mcptypes.ResourceTemplate]:
        return []

    return server


def _create_mcp_server(
    agent: AnyAgent,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> uvicorn.Server:
    root = endpoint.lstrip("/").rstrip("/")
    msg_endpoint = f"/{root}/messages/"
    sse = SseServerTransport(msg_endpoint)
    server = _create_mcp_server_instance(agent)
    init_options = server.create_initialization_options()

    async def _handle_sse(request: Request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], init_options)
        # Return empty response to avoid NoneType error
        return Response()

    routes = [
        Route(f"/{root}/sse", endpoint=_handle_sse, methods=["GET"]),
        Mount(msg_endpoint, app=sse.handle_post_message),
    ]
    starlette_app = Starlette(routes=routes)
    config = uvicorn.Config(starlette_app, host=host, port=port, log_level=log_level)
    return uvicorn.Server(config)


async def serve_mcp_async(
    agent: AnyAgent,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> tuple[asyncio.Task[Any], uvicorn.Server]:
    """Provide an A2A server to be used in an event loop."""
    uv_server = _create_mcp_server(agent, host, port, endpoint, log_level)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    return (task, uv_server)


async def serve_a2a_async(
    server: A2AStarletteApplication,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> tuple[asyncio.Task[Any], uvicorn.Server]:
    """Provide an A2A server to be used in an event loop."""
    uv_server = _create_server(server, host, port, endpoint, log_level)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    if port == 0:
        server_port = uv_server.servers[0].sockets[0].getsockname()[1]
        server.agent_card.url = f"http://{host}:{server_port}/{endpoint.lstrip('/')}"
    return (task, uv_server)


def serve_a2a(
    server: A2AStarletteApplication,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
    server_queue: Queue[int] | None = None,
) -> None:
    """Serve the A2A server."""

    # Note that the task should be kept somewhere
    # because the loop only keeps weak refs to tasks
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    async def run() -> None:
        (task, uv_server) = await serve_a2a_async(
            server, host, port, endpoint, log_level
        )
        if server_queue:
            server_queue.put(uv_server.servers[0].sockets[0].getsockname()[1])
        await task

    return run_async_in_sync(run())
