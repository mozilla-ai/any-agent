from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from .agent_card import _get_agent_card
from .agent_executor import AnyAgentExecutor

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


def _get_a2a_server(
    agent: AnyAgent, serving_config: ServingConfig
) -> A2AStarletteApplication:
    agent_card = _get_agent_card(agent, serving_config)

    request_handler = DefaultRequestHandler(
        agent_executor=AnyAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)


def serve_a2a(server: A2AStarletteApplication, host: str, port: int) -> None:
    """Serve the A2A server."""
    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)
