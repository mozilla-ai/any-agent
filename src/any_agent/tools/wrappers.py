import importlib
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from any_agent.config import AgentFramework, MCPTool, Tool
from any_agent.tools.mcp import (
    AgnoMCPServerStdio,
    GoogleMCPServerStdio,
    LangchainMCPServerStdio,
    LlamaIndexMCPServerStdio,
    MCPServerBase,
    OpenAIMCPServerStdio,
    SmolagentsMCPServerStdio,
)


def wrap_tool_openai(tool: Tool) -> Any:
    from agents import Tool as AgentTool
    from agents import function_tool

    if not isinstance(tool, AgentTool):
        return function_tool(tool)
    return tool


def wrap_tool_langchain(tool: Tool) -> Any:
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as langchain_tool

    if not isinstance(tool, BaseTool):
        return langchain_tool(tool)  # type: ignore[arg-type]
    return tool


def wrap_tool_smolagents(tool: Tool) -> Any:
    from smolagents import Tool
    from smolagents import tool as smolagents_tool

    if not isinstance(tool, Tool):
        return smolagents_tool(tool)
    return tool


def wrap_tool_llama_index(tool: Tool) -> Any:
    from llama_index.core.tools import FunctionTool

    if not isinstance(tool, FunctionTool):
        return FunctionTool.from_defaults(tool)  # type: ignore[arg-type]
    return tool


def wrap_tool_google(tool: Tool) -> Any:
    from google.adk.tools import BaseTool, FunctionTool

    if not isinstance(tool, BaseTool):
        return FunctionTool(tool)
    return tool


def wrap_tool_agno(tool: Tool) -> Any:
    # Agno lets you pass callables directly in as tools ❤️
    return tool


async def wrap_mcp_server(
    mcp_tool: MCPTool,
    agent_framework: AgentFramework,
) -> MCPServerBase:
    """Generic MCP server wrapper that can work with different frameworks
    based on the specified agent_framework
    """
    # Select the appropriate manager based on agent_framework
    mcp_server_map: dict[AgentFramework, type[MCPServerBase]] = {
        AgentFramework.OPENAI: OpenAIMCPServerStdio,
        AgentFramework.SMOLAGENTS: SmolagentsMCPServerStdio,
        AgentFramework.LANGCHAIN: LangchainMCPServerStdio,
        AgentFramework.GOOGLE: GoogleMCPServerStdio,
        AgentFramework.LLAMAINDEX: LlamaIndexMCPServerStdio,
        AgentFramework.AGNO: AgnoMCPServerStdio,
    }

    if agent_framework not in mcp_server_map:
        raise NotImplementedError(
            f"Unsupported agent type: {agent_framework}. Currently supported types are: {mcp_server_map.keys()}",
        )

    # Create the manager instance which will manage the MCP tool context
    manager_class = mcp_server_map[agent_framework]
    manager = manager_class(mcp_tool)
    await manager.setup_tools()

    return manager


WRAPPERS: dict[AgentFramework, Callable[..., Any]] = {
    AgentFramework.GOOGLE: wrap_tool_google,
    AgentFramework.OPENAI: wrap_tool_openai,
    AgentFramework.LANGCHAIN: wrap_tool_langchain,
    AgentFramework.SMOLAGENTS: wrap_tool_smolagents,
    AgentFramework.LLAMAINDEX: wrap_tool_llama_index,
    AgentFramework.AGNO: wrap_tool_agno,
}


async def import_and_wrap_tools(
    tools: Sequence[Tool],
    agent_framework: AgentFramework,
) -> tuple[list[Tool], list[MCPServerBase]]:
    wrapper = WRAPPERS[agent_framework]

    wrapped_tools = list[Tool]()
    mcp_servers = list[MCPServerBase]()
    for tool in tools:
        if isinstance(tool, MCPTool):
            # MCP adapters are usually implemented as context managers.
            # We wrap the server using `MCPServerBase` so the
            # tools can be used as any other callable.
            mcp_server = await wrap_mcp_server(tool, agent_framework)
            mcp_servers.append(mcp_server)
        elif isinstance(tool, str):
            module, func = tool.rsplit(".", 1)
            imported_module = importlib.import_module(module)
            imported_tool = getattr(imported_module, func)
            if inspect.isclass(imported_tool):
                imported_tool = imported_tool()
            wrapped_tools.append(wrapper(imported_tool))
        elif callable(tool):
            wrapped_tools.append(wrapper(tool))
        else:
            raise ValueError(
                f"Tool {tool} needs to be of type `MCPTool`, `str` or `callable` but is {type(tool)}",
            )

    return wrapped_tools, mcp_servers
