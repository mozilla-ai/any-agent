from typing import TYPE_CHECKING, Union
from .langchain import run_langchain_agent
from .openai import run_openai_agent
from .smolagents import run_smolagents_agent


if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph
    from agents import Agent
    from smolagents import AgentType


def isinstance_without_import(instance, module, name):
    for cls in type(instance).mro():
        if (cls.__module__, cls.__name__) == (module, name):
            return True
    return False


def run_agent(
    agent: Union["CompiledGraph", "Agent", "AgentType"],
    query: str,
):
    if isinstance_without_import(agent, "langchain.graph", "CompiledGraph"):
        return run_langchain_agent(agent, query)
    if isinstance_without_import(agent, "agents.agent", "Agent"):
        return run_openai_agent(agent, query)
    if isinstance_without_import(agent, "smolagents", "AgentType"):
        return run_smolagents_agent(agent, query)
    else:
        raise NotImplementedError(f"{agent} is not supported yet.")
