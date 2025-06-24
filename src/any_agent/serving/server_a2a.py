from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from starlette.routing import Mount

from any_agent.serving.task_manager import TaskManager
from any_agent.utils import run_async_in_sync

from .agent_card import _get_agent_card
from .agent_executor import AnyAgentExecutor
from .envelope import prepare_agent_for_a2a, prepare_agent_for_a2a_async

if TYPE_CHECKING:
    from multiprocessing import Queue

    from any_agent import AnyAgent
    from any_agent.serving import A2AServingConfig


def _get_a2a_app(
    agent: AnyAgent, serving_config: A2AServingConfig
) -> A2AStarletteApplication:
    agent = prepare_agent_for_a2a(agent)

    agent_card = _get_agent_card(agent, serving_config)
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

    agent_card = _get_agent_card(agent, serving_config)
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
