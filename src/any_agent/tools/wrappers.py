import inspect
import importlib
from collections.abc import Callable
from textwrap import dedent

from any_agent.schema import AgentFramework, MCPTool
from any_agent.tools.mcp import SmolagentsMCPToolsManager


def import_and_wrap_tools(
    tools: list[str | dict], agent_framework: AgentFramework
) -> list[Callable]:
    # Select the appropriate wrapper based on agent_framework
    wrapper_map = {
        AgentFramework.OPENAI: wrap_tool_openai,
        AgentFramework.LANGCHAIN: wrap_tool_langchain,
        AgentFramework.SMOLAGENTS: wrap_tool_smolagents,
    }

    wrapper = wrapper_map[agent_framework]

    imported_tools = []
    for tool in tools:
        if isinstance(tool, MCPTool):  # Handle MCP tool configuration
            # This is an MCP tool definition
            mcp_tools = wrap_mcp_server(tool, agent_framework)
            imported_tools.extend(mcp_tools)
        else:  # Regular string tool reference
            module, func = tool.rsplit(".", 1)
            module = importlib.import_module(module)
            imported_tool = getattr(module, func)
            if inspect.isclass(imported_tool):
                imported_tool = imported_tool()
            imported_tools.append(wrapper(imported_tool))
    return imported_tools


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


def wrap_mcp_server(mcp_tool: MCPTool, agent_framework: AgentFramework):
    """
    Generic MCP server wrapper that can work with different frameworks
    based on the specified agent_framework
    """
    if agent_framework != AgentFramework.SMOLAGENTS:
        raise NotImplementedError(
            f"Unsupported agent type: {agent_framework}. Currently only smolagents is supported for MCP server tools."
        )
    # Create the manager instance which will manage the MCP tool context
    manager = SmolagentsMCPToolsManager(mcp_tool)

    # Get all tools from the manager
    tools = manager.tools

    # Only add the tools listed in mcp_tool['tools'] if specified
    requested_tools = mcp_tool.tools
    filtered_tools = [tool for tool in tools if tool.name in requested_tools]
    if len(filtered_tools) != len(requested_tools):
        tool_names = [tool.name for tool in filtered_tools]
        raise ValueError(
            dedent(f"""Could not find all requested tools in the MCP server:
                         Requested: {requested_tools}
                         Set:   {tool_names}""")
        )

    return filtered_tools
    return tools
