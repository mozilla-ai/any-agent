import inspect
import importlib
from collections.abc import Callable
import os
from textwrap import dedent


def import_and_wrap_tools(tools: list[str], wrapper: Callable) -> list[Callable]:
    imported_tools = []
    for tool in tools:
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


# Global registry to keep manager instances alive
_mcp_managers = {}


class MCP_ToolsManager:
    def __init__(self, mcp_tool: dict):
        from mcp import StdioServerParameters
        from smolagents import ToolCollection

        # Generate a unique identifier for this manager instance
        self.id = id(self)

        self.server_parameters = StdioServerParameters(
            command=mcp_tool["command"],
            args=mcp_tool["args"],
            env={**os.environ},
        )

        # Store the context manager itself
        self.context = ToolCollection.from_mcp(self.server_parameters)
        # Enter the context
        self.tool_collection = self.context.__enter__()
        self.tools = self.tool_collection.tools

        # Register self in the global registry to prevent garbage collection
        _mcp_managers[self.id] = self

    def __del__(self):
        # Exit the context when the class is deleted/garbage collected
        if hasattr(self, "context") and self.context:
            try:
                self.context.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing MCP context: {e}")

        # Remove from registry
        if hasattr(self, "id") and self.id in _mcp_managers:
            del _mcp_managers[self.id]


def wrap_mcp_server_smolagents(mcp_tool: dict):
    # Create the manager instance which will manage the MCP tool context
    # The manager will be kept alive in the _mcp_managers registry
    manager = MCP_ToolsManager(mcp_tool)

    # Return the tools along with a reference to the manager's ID
    # The returned tools are now associated with their manager
    # which will stay alive in the global registry
    tools = manager.tools

    # only add the tools listed in mcp_tool['tools']. Throw an error if a requested tool isn't found
    if "tools" in mcp_tool:
        tools = [tool for tool in tools if tool.name in mcp_tool["tools"]]
        if len(tools) != len(mcp_tool["tools"]):
            tool_names = [tool.name for tool in tools]
            raise ValueError(
                dedent(f"""Could not find all requested tools in the MCP server:
                             Requested: {mcp_tool['tools']}
                             Set:   {tool_names}""")
            )

    return tools


def cleanup_mcp_managers():
    """Manually clean up all MCP managers"""
    global _mcp_managers
    managers = list(_mcp_managers.values())
    for manager in managers:
        del manager
    _mcp_managers.clear()
    print(f"Cleaned up all MCP managers, current count: {len(_mcp_managers)}")
