from typing import TYPE_CHECKING, Union

from .langchain import load_lanchain_agent
from .openai import load_openai_agent
from .smolagents import load_smolagents_agent


if TYPE_CHECKING:
    from any_agent.schema import AgentSchema
    from langgraph.graph.graph import CompiledGraph
    from agents import Agent
    from smolagents import AgentType


def load_agent(
    framework: str,
    main_agent: "AgentSchema",
    managed_agents: list["AgentSchema"] | None = None,
) -> Union["CompiledGraph", "Agent", "AgentType"]:
    """Loads an agent using the provided `framework`.

    The loaded agent can be then passed to [`run_agent`][any_agent.runners.run_agent]

    Args:
        main_agent: This will be the returned agent.
        managed_agents: A list of agents that will be managed by `main_agent`.

    Returns:
        An agent instance configured according to the given `AgentSchema`.
        The instance type depends on the selected framework.
    """
    match framework:
        case "langchain":
            return load_lanchain_agent(main_agent, managed_agents)
        case "openai":
            return load_openai_agent(main_agent, managed_agents)
        case "smolagents":
            return load_smolagents_agent(main_agent, managed_agents)
        case _:
            raise NotImplementedError(f"{framework} is not supported yet.")
