from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import mcp.types as mcptypes
import uvicorn
from mcp.server import Server as MCPServer
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from any_agent.serving.server_handle import ServerHandle

if TYPE_CHECKING:
    from starlette.requests import Request

    from any_agent import AnyAgent


def _create_mcp_server_instance(agent: AnyAgent) -> MCPServer[Any]:
    server = MCPServer[Any]("any-agent-mcp-server")

    @server.list_tools()  # type: ignore[no-untyped-call,misc]
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

    @server.call_tool()  # type: ignore[misc]
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


def _get_mcp_app(
    agent: AnyAgent,
    endpoint: str,
) -> Starlette:
    """Provide an MCP server to be used in an event loop."""
    root = endpoint.lstrip("/").rstrip("/")
    msg_endpoint = f"/{root}/messages/"
    sse = SseServerTransport(msg_endpoint)
    server = _create_mcp_server_instance(agent)
    init_options = server.create_initialization_options()

    async def _handle_sse(request: Request):  # type: ignore[no-untyped-def]
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
    return Starlette(routes=routes)


async def serve_mcp_async(
    agent: AnyAgent,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> ServerHandle:
    """Provide an MCP server to be used in an event loop."""
    config = uvicorn.Config(
        _get_mcp_app(agent, endpoint), host=host, port=port, log_level=log_level
    )
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    return ServerHandle(task=task, server=uv_server)
