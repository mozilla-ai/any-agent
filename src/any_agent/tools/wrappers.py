import inspect
import importlib
from collections.abc import Callable


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
