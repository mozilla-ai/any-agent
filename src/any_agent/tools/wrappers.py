import inspect
from collections.abc import Callable
from functools import wraps

from any_agent.config import AgentFramework, MCPTool
from any_agent.tools.mcp import (
    AgnoMCPServerStdio,
    GoogleMCPServerStdio,
    LangchainMCPServerStdio,
    LlamaIndexMCPServerStdio,
    MCPServerBase,
    OpenAIMCPServerStdio,
    SmolagentsMCPServerStdio,
)


def wrap_tool_openai(tool):
    from agents import Tool, function_tool

    if not isinstance(tool, Tool):
        return function_tool(tool)
    return tool


def wrap_tool_langchain(tool):
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as langchain_tool

    if not isinstance(tool, BaseTool):
        return langchain_tool(tool)
    return tool


def wrap_tool_smolagents(tool):
    from smolagents import Tool
    from smolagents import tool as smolagents_tool

    if not isinstance(tool, Tool):
        # this wrapping needed until https://github.com/huggingface/smolagents/pull/1203 is merged and released
        @wraps(tool)
        def wrapped_function(*args, **kwargs):
            return tool(*args, **kwargs)

        return smolagents_tool(wrapped_function)
    return tool


def wrap_tool_llama_index(tool):
    from llama_index.core.tools import FunctionTool

    if not isinstance(tool, FunctionTool):
        return FunctionTool.from_defaults(tool)
    return tool


def wrap_tool_google(tool):
    from google.adk.tools import BaseTool, FunctionTool

    if not isinstance(tool, BaseTool):
        return FunctionTool(tool)
    return tool


def wrap_tool_agno(tool):
    # Agno lets you pass callables directly in as tools ❤️
    return tool


async def wrap_mcp_server(
    mcp_tool: MCPTool, agent_framework: AgentFramework
) -> MCPServerBase:
    """
    Generic MCP server wrapper that can work with different frameworks
    based on the specified agent_framework
    """
    # Select the appropriate manager based on agent_framework
    mcp_server_map = {
        AgentFramework.OPENAI: OpenAIMCPServerStdio,
        AgentFramework.SMOLAGENTS: SmolagentsMCPServerStdio,
        AgentFramework.LANGCHAIN: LangchainMCPServerStdio,
        AgentFramework.GOOGLE: GoogleMCPServerStdio,
        AgentFramework.LLAMAINDEX: LlamaIndexMCPServerStdio,
        AgentFramework.AGNO: AgnoMCPServerStdio,
    }

    if agent_framework not in mcp_server_map:
        msg = f"Unsupported agent type: {agent_framework}. Currently supported types are: {mcp_server_map.keys()}"
        raise NotImplementedError(msg)

    # Create the manager instance which will manage the MCP tool context
    manager_class = mcp_server_map[agent_framework]
    manager: MCPServerBase = manager_class(mcp_tool)
    await manager.setup_tools()

    return manager


WRAPPERS = {
    AgentFramework.GOOGLE: wrap_tool_google,
    AgentFramework.OPENAI: wrap_tool_openai,
    AgentFramework.LANGCHAIN: wrap_tool_langchain,
    AgentFramework.SMOLAGENTS: wrap_tool_smolagents,
    AgentFramework.LLAMAINDEX: wrap_tool_llama_index,
    AgentFramework.AGNO: wrap_tool_agno,
}


def verify_callable(tool: Callable) -> None:
    """
    Verify a few things about the callable:
    - It needs to have some sort of docstring that describes what it does
    - It needs to have typed argument
    - It needs to have a typed return

    We need these things because this info gets provided to the agent so that they know how and when to call the tool.
    """
    signature = inspect.signature(tool)
    if not tool.__doc__:
        msg = f"Tool {tool} needs to have a docstring but does not"
        raise ValueError(msg)

    # Check if the function has a return type
    if signature.return_annotation is inspect.Signature.empty:
        msg = f"Tool {tool} needs to have a return type but does not"
        raise ValueError(msg)
    # Check if all parameters have type annotations
    for param in signature.parameters.values():
        if param.annotation is inspect.Signature.empty:
            msg = f"Tool {tool} needs to have typed arguments but does not"
            raise ValueError(msg)


async def wrap_tools(
    tools: list[Callable, MCPTool], agent_framework: AgentFramework
) -> tuple[list[Callable], list[MCPServerBase]]:
    wrapper = WRAPPERS[agent_framework]

    wrapped_tools = []
    mcp_servers = []
    for tool in tools:
        if isinstance(tool, MCPTool):
            # MCP adapters are usually implemented as context managers.
            # We wrap the server using `MCPServerBase` so the
            # tools can be used as any other callable.
            mcp_server = await wrap_mcp_server(tool, agent_framework)
            mcp_servers.append(mcp_server)
        elif callable(tool):
            verify_callable(tool)
            wrapped_tools.append(wrapper(tool))
        else:
            msg = f"Tool {tool} needs to be of type `MCPTool`, `str` or `callable` but is {type(tool)}"
            raise ValueError(msg)

    return wrapped_tools, mcp_servers
