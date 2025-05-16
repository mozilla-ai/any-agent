from __future__ import annotations

import asyncio
from asyncio import Task
from typing import TYPE_CHECKING, Any

from common.server import A2AServer

from .agent_card import _get_agent_card
from .task_manager import AnyAgentTaskManager

if TYPE_CHECKING:
    import uvicorn

    from any_agent import AnyAgent
    from any_agent.config import ServingConfig

AGENT_CARD_NOT_DEFINED = "agent_card is not defined"
REQUEST_HANDLER_NOT_DEFINED = "request_handler is not defined"


class ServerNotStartingError(Exception):
    """Raised when the underlying uvicorn server did not start within the expected time."""


class A2AServerAsync(A2AServer):
    """Variant of A2AServer that serves an agent within an event loop."""

    def __init__(self, **kwargs: Any) -> None:
        """Build instance from parent constructor."""
        super().__init__(**kwargs)  # python3.0+
        self._server: uvicorn.Server | None = None
        self._server_task: Task[Any] | None = None

    async def serve(self) -> None:
        """Start the A2AServerAsync within an event loop."""
        if self.agent_card is None:
            raise ValueError(AGENT_CARD_NOT_DEFINED)

        if self.task_manager is None:
            raise ValueError(REQUEST_HANDLER_NOT_DEFINED)

        import uvicorn

        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:  # noqa: ASYNC110
            # TODO check if events could be provided inside uvicorn to notify
            # lifecycle events
            await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        """Stop the asynchronous server."""
        if self._server:
            await self._server.shutdown()
        if self._task:
            self._task.cancel()

    def uvi_server(self) -> uvicorn.Server | None:
        """Return the underlying uvicorn server serving the agent."""
        return self._server


def _get_a2a_server(agent: AnyAgent, serving_config: ServingConfig) -> A2AServer:
    return A2AServer(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )


def _get_a2a_server_async(
    agent: AnyAgent, serving_config: ServingConfig
) -> A2AServerAsync:
    return A2AServerAsync(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )
