from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from any_agent import AgentFramework

if TYPE_CHECKING:
    from any_agent import AnyAgent
    from any_agent.serving.a2a.config_a2a import A2AServingConfig


def _get_agent_card(agent: AnyAgent, serving_config: A2AServingConfig) -> AgentCard:
    skills = serving_config.skills
    if skills is None:
        skills = []
        for tool in agent._tools:
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
                    tags=[],
                )
            )
    if agent.config.description is None:
        msg = "Agent description is not set. Please set the `description` field in the `AgentConfig`."
        raise ValueError(msg)
    endpoint = serving_config.endpoint.lstrip("/")
    return AgentCard(
        name=agent.config.name,
        description=agent.config.description,
        version=serving_config.version,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        url=f"http://{serving_config.host}:{serving_config.port}/{endpoint}",
        capabilities=AgentCapabilities(
            streaming=False, pushNotifications=True, stateTransitionHistory=False
        ),
        skills=skills,
    )
