import os
from typing import Optional

from loguru import logger

from any_agent.schema import AgentSchema
from any_agent.instructions import get_instructions
from any_agent.tools.wrappers import import_and_wrap_tools, wrap_tool_openai


try:
    from agents import (
        Agent,
        AsyncOpenAI,
        OpenAIChatCompletionsModel,
    )

    agents_available = True
except ImportError:
    agents_available = None


def _get_model(agent_config: AgentSchema):
    if agent_config.api_key_var and agent_config.api_base:
        external_client = AsyncOpenAI(
            api_key=os.environ[agent_config.api_key_var],
            base_url=agent_config.api_base,
        )
        return OpenAIChatCompletionsModel(
            model=agent_config.model_id,
            openai_client=external_client,
        )
    return agent_config.model_id


@logger.catch(reraise=True)
def load_openai_agent(
    main_agent: AgentSchema,
    managed_agents: Optional[list[AgentSchema]] = None,
) -> Agent:
    """Loads an agent using `openai-agents`.

    Args:
        main_agent: This will be the returned agent.
            See [`AgentSchema`][].
        managed_agents: A list of agents that will be managed by `main_agent`.
            In `openai-agents`, the managed agents can be used either
            [as a tool](https://openai.github.io/openai-agents-python/tools/#agents-as-tools)
            or as a [handoff](https://openai.github.io/openai-agents-python/handoffs/).

            To indicate that an agent should be used as a `handoff`, you can set the
            `handoff` argument to `True` in the corresponding `AgentSchema`.

    Returns:
        An instance of `agents.Agent` configured according to the given `AgentSchema`.

    Raises:
        ImportError: If `openai-agents` is not installed.

    Examples:

        >>> agent = load_openai_agent(AgentSchema(model_id="o3-mini"))
    """
    if not agents_available:
        raise ImportError("You need to `pip install openai-agents` to use this agent")

    if not managed_agents and not main_agent.tools:
        main_agent.tools = [
            "any_agent.tools.search_web",
            "any_agent.tools.visit_webpage",
        ]
    tools = import_and_wrap_tools(main_agent.tools, wrap_tool_openai)

    handoffs = []
    if managed_agents:
        for managed_agent in managed_agents:
            instance = Agent(
                name=managed_agent.name,
                instructions=get_instructions(managed_agent.instructions),
                model=_get_model(managed_agent),
                tools=import_and_wrap_tools(managed_agent.tools, wrap_tool_openai),
            )
            if managed_agent.handoff:
                handoffs.append(instance)
            else:
                tools.append(
                    instance.as_tool(
                        tool_name=instance.name,
                        tool_description=managed_agent.description
                        or f"Use the agent: {managed_agent.name}",
                    )
                )

    return Agent(
        name=main_agent.name,
        instructions=main_agent.instructions,
        model=_get_model(main_agent),
        handoffs=handoffs,
        tools=tools,
    )
