from __future__ import annotations

from typing import TYPE_CHECKING

from common.server import A2AServer

from .agent_card import _get_agent_card
from .task_manager import AnyAgentTaskManager

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig

class A2AServerAsync(A2AServer):
    async def start_async(self):
        if self.agent_card is None:
            raise ValueError('agent_card is not defined')

        if self.task_manager is None:
            raise ValueError('request_handler is not defined')

        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


def _get_a2a_server(agent: AnyAgent, serving_config: ServingConfig) -> A2AServer:
    return A2AServer(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )

def _get_a2a_server_async(agent: AnyAgent, serving_config: ServingConfig) -> A2AServerAsync:
    return A2AServerAsync(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )
