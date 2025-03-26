import os
from typing import Optional

from loguru import logger

from any_agent.schema import AgentSchema
from any_agent.tools.wrappers import import_and_wrap_tools, wrap_tool_smolagents


try:
    import smolagents

    smolagents_available = True
except ImportError:
    smolagents_available = None

DEFAULT_AGENT_CLASS = "ToolCallingAgent"
DEFAULT_MODEL_CLASS = "LiteLLMModel"


def _get_model(agent_config: AgentSchema):
    model_class = getattr(smolagents, agent_config.model_class or DEFAULT_MODEL_CLASS)
    kwargs = {
        "model_id": agent_config.model_id,
    }
    if agent_config.api_base:
        kwargs["api_base"] = agent_config.api_base
    if agent_config.api_key_var:
        kwargs["api_key"] = os.environ[agent_config.api_key_var]
    return model_class(**kwargs)


@logger.catch(reraise=True)
def load_smolagents_agent(
    main_agent: AgentSchema,
    managed_agents: Optional[list[AgentSchema]] = None,
) -> smolagents.AgentType:
    if not smolagents_available:
        raise ImportError("You need to `pip install smolagents` to use this agent")

    if not managed_agents and not main_agent.tools:
        main_agent.tools = [
            "any_agent.tools.search_web",
            "any_agent.tools.visit_webpage",
        ]
    tools = import_and_wrap_tools(main_agent.tools, wrap_tool_smolagents)

    managed_agents_instanced = []
    if managed_agents:
        for managed_agent in managed_agents:
            agent_class = getattr(
                smolagents, managed_agent.agent_class or DEFAULT_AGENT_CLASS
            )
            managed_agents_instanced.append(
                agent_class(
                    name=managed_agent.name,
                    model=_get_model(managed_agent),
                    tools=import_and_wrap_tools(
                        managed_agent.tools, wrap_tool_smolagents
                    ),
                    description=managed_agent.description
                    or f"Use the agent: {managed_agent.name}",
                )
            )

    main_agent_class = getattr(
        smolagents, main_agent.agent_class or DEFAULT_AGENT_CLASS
    )
    main_agent_instance = main_agent_class(
        name=main_agent.name,
        model=_get_model(main_agent),
        tools=tools,
        managed_agents=managed_agents_instanced,
    )

    return main_agent_instance
