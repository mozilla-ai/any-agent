from collections.abc import Callable, Iterable
from typing import Any

from pydantic import TypeAdapter

from any_agent.config import AgentFramework, Tool

from .agno import AgnoTool
from .google import GoogleTool
from .langchain import LangchainTool
from .llama_index import LlamaIndexTool
from .openai import OpenAITool
from .smolagents import SmolagentsTool
from .tinyagent import TinyAgentTool

AnyTool = (
    AgnoTool
    | GoogleTool
    | LangchainTool
    | OpenAITool
    | SmolagentsTool
    | TinyAgentTool
    | LlamaIndexTool
)


def _wrap_tool(tool: Tool, agent_framework: AgentFramework) -> AnyTool:
    return TypeAdapter[AnyTool](AnyTool).validate_python(
        {"tool": tool, "framework": agent_framework}
    )


async def wrap_tools(
    tools: Iterable[Tool | Callable[..., Any]], agent_framework: AgentFramework
) -> list[AnyTool]:
    """Wrap a list of tools for the specified framework."""
    return [_wrap_tool(tool, agent_framework) for tool in tools]


__all__ = [
    "AgnoTool",
    "GoogleTool",
    "LangchainTool",
    "LlamaIndexTool",
    "OpenAITool",
    "SmolagentsTool",
    "TinyAgentTool",
    "Tool",
    "wrap_tools",
]
