from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from common.types import AgentCapabilities, AgentCard, AgentSkill

from any_agent import AgentFramework

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.config import ServingConfig


def _get_agent_card(agent: AnyAgent, serving_config: ServingConfig) -> AgentCard:
    skills = []
    for tool in agent._main_agent_tools:
        if hasattr(tool, "name"):
            tool_name = tool.name
            tool_description = tool.description
        elif agent.framework is AgentFramework.LLAMA_INDEX:
            tool_name = tool.metadata.name
            tool_description = tool.metadata.description
        else:
            tool_name = tool.__name__
            tool_description = inspect.getdoc(tool)
        skills.append(
            AgentSkill(
                id=f"{agent.config.name}-{tool_name}",
                name=tool_name,
                description=tool_description,
            )
        )
    return AgentCard(
        name=agent.config.name,
        description=agent.config.description,
        version=serving_config.version,
        url=f"http://{serving_config.host}:{serving_config.port}/",
        capabilities=AgentCapabilities(
            streaming=False, pushNotifications=False, stateTransitionHistory=False
        ),
        skills=skills,
    )
