from __future__ import annotations

from typing import TYPE_CHECKING, Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from starlette.routing import Mount

from .agent_card import _get_agent_card
from .agent_executor import AnyAgentExecutor

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig

import asyncio


def _get_a2a_app(
    agent: AnyAgent, serving_config: ServingConfig
) -> A2AStarletteApplication:
    agent_card = _get_agent_card(agent, serving_config)

    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent),
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
    return (task, uv_server)


def serve_a2a(
    server: A2AStarletteApplication,
    host: str,
    port: int,
    endpoint: str,
    log_level: str = "warning",
) -> None:
    """Serve the A2A server."""

    # Note that the task should be kept somewhere
    # because the loop only keeps weak refs to tasks
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    async def run() -> None:
        (task, _) = await serve_a2a_async(server, host, port, endpoint, log_level)
        await task

    return asyncio.get_event_loop().run_until_complete(run())
