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
from pydantic import BaseModel
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
    from any_agent.serving import A2AServingConfig, MCPServingConfig


def _create_mcp_server_instance(agent: AnyAgent) -> MCPServer[Any]:
    server = MCPServer("any-agent-mcp-server", version="0.1.0")

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
        name: str, arguments: dict[str, Any]
    ) -> list[mcptypes.TextContent | mcptypes.ImageContent | mcptypes.EmbeddedResource]:
        result = await agent.run_async(arguments["query"])
        output = result.final_output
        if isinstance(output, BaseModel):
            serialized_output = output.model_dump_json()
        else:
            serialized_output = str(output)
        return [mcptypes.TextContent(type="text", text=serialized_output)]

    return server


async def serve_async(
    starlette_app, host, port, log_level
) -> tuple[asyncio.Task[Any], uvicorn.Server]:
    config = uvicorn.Config(starlette_app, host=host, port=port, log_level=log_level)
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    return (task, uv_server)


async def serve_mcp_async(
    agent: AnyAgent,
    serving_config: A2AServingConfig,
) -> tuple[asyncio.Task[Any], uvicorn.Server]:
    """Provide an MCP server to be used in an event loop."""
    root = serving_config.endpoint.lstrip("/").rstrip("/")
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
        # Please check https://github.com/modelcontextprotocol/python-sdk/blob/1eb1bba83c70c3121bce7fc0263e5fac2c3f0520/src/mcp/server/sse.py#L33
        return Response()

    routes = [
        Route(f"/{root}/sse", endpoint=_handle_sse, methods=["GET"]),
        Mount(msg_endpoint, app=sse.handle_post_message),
    ]
    starlette_app = Starlette(routes=routes)

    return await serve_async(
        starlette_app,
        serving_config.host,
        serving_config.port,
        serving_config.log_level,
    )


async def serve_a2a_async(
    agent: AnyAgent,
    serving_config: A2AServingConfig,
) -> tuple[asyncio.Task[Any], uvicorn.Server]:
    """Provide an A2A server to be used in an event loop."""
    agent_card = _build_agent_card(agent, serving_config)
    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
    )
    app_builder = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    root = serving_config.endpoint.lstrip("/").rstrip("/")
    starlette_app = app_builder.build()
    internal_router = Starlette(routes=[Mount(f"/{root}", routes=starlette_app.routes)])

    return await serve_async(
        internal_router,
        serving_config.host,
        serving_config.port,
        serving_config.log_level,
    )


def serve_a2a(
    agent: AnyAgent,
    serving_config: A2AServingConfig,
) -> None:
    """Serve the A2A server."""

    # Note that the task should be kept somewhere
    # because the loop only keeps weak refs to tasks
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    async def run() -> None:
        (task, _) = await serve_a2a_async(agent, serving_config)
        await task

    return run_async_in_sync(run())


def serve_mcp(
    agent: AnyAgent,
    serving_config: MCPServingConfig,
) -> None:
    """Serve the MCP server."""

    # Note that the task should be kept somewhere
    # because the loop only keeps weak refs to tasks
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    async def run() -> None:
        (task, _) = await serve_mcp_async(agent, serving_config)
        await task

    return run_async_in_sync(run())
