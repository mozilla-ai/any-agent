from __future__ import annotations

from typing import TYPE_CHECKING

from common.server import A2AServer
import threading
import asyncio
from asyncio import sleep as asleep
import uvicorn
from any_agent.logging import logger

from .agent_card import _get_agent_card
from .task_manager import AnyAgentTaskManager

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


class ServerNotStartingError(Exception):
    "Raised when the underlying uvicorn server did not start within the expected time"
    pass

class A2AServerThreaded(A2AServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # python3.0+
        self._server = None
        self._thread = None

    async def start_threaded(self):
        if self.agent_card is None:
            raise ValueError('agent_card is not defined')

        if self.task_manager is None:
            raise ValueError('request_handler is not defined')

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run)
        self._thread.start()
        while not self._server.started:
            await asyncio.sleep(0.01)
        # TODO Wait for thread


    def join(self) -> None:
        self._thread.join()

    def get_server(self) -> uvicorn.Server:
        return self._server
    
def _get_a2a_server(agent: AnyAgent, serving_config: ServingConfig) -> A2AServer:
    return A2AServer(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )

def _get_a2a_server_threaded(agent: AnyAgent, serving_config: ServingConfig) -> A2AServerThreaded:
    return A2AServerThreaded(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )
