from typing import TYPE_CHECKING
from .langchain import load_lanchain_agent
from .openai import load_openai_agent
from .smolagents import load_smolagents_agent


if TYPE_CHECKING:
    from any_agent.schema import AgentSchema


def load_agent(
    framework: str,
    main_agent: "AgentSchema",
    managed_agents: list["AgentSchema"] | None = None,
):
    match framework:
        case "langchain":
            return load_lanchain_agent(main_agent, managed_agents)
        case "openai":
            return load_openai_agent(main_agent, managed_agents)
        case "smolagents":
            return load_smolagents_agent(main_agent, managed_agents)
        case _:
            raise NotImplementedError(f"{framework} is not supported yet.")
