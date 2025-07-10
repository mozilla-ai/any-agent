from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from starlette.applications import Starlette
from starlette.routing import Mount

from any_agent.serving.a2a.context_manager import ContextManager
from any_agent.serving.server_handle import ServerHandle

from .agent_card import _get_agent_card
from .agent_executor import AnyAgentExecutor
from .envelope import prepare_agent_for_a2a_async

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.serving import A2AServingConfig


async def _get_a2a_app_async(
    agent: AnyAgent, serving_config: A2AServingConfig
) -> A2AStarletteApplication:
    agent = await prepare_agent_for_a2a_async(agent)

    agent_card = _get_agent_card(agent, serving_config)
    task_manager = ContextManager(serving_config)
    push_notification_config_store = serving_config.push_notifier_store_type()
    push_notification_sender = serving_config.push_notifier_sender_type(
        httpx_client=httpx.AsyncClient(),  # type: ignore[call-arg]
        config_store=push_notification_config_store,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent, task_manager),
        task_store=serving_config.task_store_type(),
        push_config_store=push_notification_config_store,
        push_sender=push_notification_sender,
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
) -> ServerHandle:
    """Provide an A2A server to be used in an event loop."""
    uv_server = _create_server(server, host, port, endpoint, log_level)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    if port == 0:
        server_port = uv_server.servers[0].sockets[0].getsockname()[1]
        server.agent_card.url = f"http://{host}:{server_port}/{endpoint.lstrip('/')}"
    return ServerHandle(task=task, server=uv_server)
