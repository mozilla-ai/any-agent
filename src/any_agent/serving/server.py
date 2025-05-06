from typing import TYPE_CHECKING

try:
    from common.server import A2AServer

    a2a_available = True
except ImportError:
    a2a_available = False

from .agent_card import _get_agent_card
from .task_manager import AnyAgentTaskManager

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


def _get_a2a_server(agent: "AnyAgent", serving_config: "ServingConfig") -> A2AServer:
    if not a2a_available:
        msg = "You need to `pip install 'any-agent[serving]'` to use this"
        raise ImportError(msg)
    return A2AServer(
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        agent_card=_get_agent_card(agent, serving_config),
        task_manager=AnyAgentTaskManager(agent),
    )
