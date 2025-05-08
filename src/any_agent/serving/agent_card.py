from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from common.types import AgentCapabilities, AgentCard, AgentSkill

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


def _get_agent_card(agent: AnyAgent, serving_config: ServingConfig) -> AgentCard:
    skills = []
    for tool in agent.config.tools:
        # TODO: handle MCP tools
        if not callable(tool):
            continue
        skills.append(
            # TODO: find what other arguments can be set.
            AgentSkill(
                id=f"{agent.config.name}-{tool.__name__}",
                name=tool.__name__,
                description=inspect.getdoc(tool),
            )
        )
    return AgentCard(
        name=agent.config.name,
        description=agent.config.description,
        version=serving_config.version,
        url=f"http://{serving_config.host}:{serving_config.port}/",
        # TODO: extend default capabilities
        capabilities=AgentCapabilities(
            streaming=False, pushNotifications=False, stateTransitionHistory=False
        ),
        skills=skills,
    )
