import inspect
import importlib
from collections.abc import Callable

from any_agent.schema import AgentFramework, MCPTool
from any_agent.tools.mcp import (
    SmolagentsMCPToolsManager,
    OpenAIMCPToolsManager,
    LangchainMCPToolsManager,
    MCPToolsManagerBase,
)


def import_and_wrap_tools(
    tools: list[str | dict], agent_framework: AgentFramework
) -> tuple[list[Callable], list[MCPToolsManagerBase]]:
    # Select the appropriate wrapper based on agent_framework
    wrapper_map = {
        AgentFramework.OPENAI: wrap_tool_openai,
        AgentFramework.LANGCHAIN: wrap_tool_langchain,
        AgentFramework.SMOLAGENTS: wrap_tool_smolagents,
    }

    wrapper = wrapper_map[agent_framework]

    imported_tools = []
    mcp_servers = []
    for tool in tools:
        if isinstance(tool, MCPTool):  # Handle MCP tool configuration
            # This is an MCP tool definition
            mcp_server = wrap_mcp_server(tool, agent_framework)
            mcp_servers.append(mcp_server)
        else:  # Regular string tool reference
            module, func = tool.rsplit(".", 1)
            module = importlib.import_module(module)
            imported_tool = getattr(module, func)
            if inspect.isclass(imported_tool):
                imported_tool = imported_tool()
            imported_tools.append(wrapper(imported_tool))
    return imported_tools, mcp_servers


def wrap_tool_openai(tool):
    from agents import function_tool, FunctionTool

    if not isinstance(tool, FunctionTool):
        return function_tool(tool)
    return tool


def wrap_tool_langchain(tool):
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as langchain_tool

    if not isinstance(tool, BaseTool):
        return langchain_tool(tool)
    return tool


def wrap_tool_smolagents(tool):
    from smolagents import Tool, tool as smolagents_tool

    if not isinstance(tool, Tool):
        return smolagents_tool(tool)

    return tool


def wrap_mcp_server(
    mcp_tool: MCPTool, agent_framework: AgentFramework
) -> MCPToolsManagerBase:
    """
    Generic MCP server wrapper that can work with different frameworks
    based on the specified agent_framework
    """
    # Select the appropriate manager based on agent_framework
    manager_map = {
        AgentFramework.OPENAI: OpenAIMCPToolsManager,
        AgentFramework.SMOLAGENTS: SmolagentsMCPToolsManager,
        AgentFramework.LANGCHAIN: LangchainMCPToolsManager,
    }

    if agent_framework not in manager_map:
        raise NotImplementedError(
            f"Unsupported agent type: {agent_framework}. Currently supported types are: {manager_map.keys()}"
        )

    # Create the manager instance which will manage the MCP tool context
    manager_class = manager_map[agent_framework]
    manager = manager_class(mcp_tool)

    return manager
