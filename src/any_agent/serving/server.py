from __future__ import annotations

from typing import TYPE_CHECKING

from common.server import A2AServer as GoogleA2AServer

from .agent_card import _get_agent_card
from .task_manager import AnyAgentTaskManager

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


class A2AServer(GoogleA2AServer):
    def __init__(
        self,
        host='0.0.0.0',
        port=5000,
        endpoint='/',
        agent_card: AgentCard = None,
        task_manager: TaskManager = None,
    ):
        super().__init__(host, port, endpoint, agent_card, task_manager)
        self._uvicorn = None

    def start(self):
        if self.agent_card is None:
            raise ValueError('agent_card is not defined')

        if self.task_manager is None:
            raise ValueError('request_handler is not defined')

        import uvicorn
        config = uvicorn.Config(self.app, port=self.port, host=self.host, log_level="info")
        self._uvicorn = uvicorn.Server(config)
        self._uvicorn.run()

    def get_uvicorn(self):
        return self._uvicorn



def _get_a2a_server(agent: AnyAgent, serving_config: ServingConfig) -> A2AServer:
    return A2AServer(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )
