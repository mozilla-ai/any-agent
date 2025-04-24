import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any

from any_agent.config import AgentFramework, MCPParams, Tool
from any_agent.tools import (
    MCPServerBase,
    get_mcp_server,
)

if TYPE_CHECKING:
    from agents import Tool as AgentTool
    from google.adk.tools import BaseTool as GoogleTool
    from langchain_core.tools import BaseTool as LangchainTool
    from llama_index.core.tools import FunctionTool as LlamaIndexTool
    from smolagents import Tool as SmolagentsTool


def wrap_tool_openai(tool: "Tool | AgentTool") -> "AgentTool":
    from agents import Tool as AgentTool
    from agents import function_tool

    if isinstance(tool, AgentTool):  # type: ignore[arg-type, misc]
        return tool  # type: ignore[return-value]

    return function_tool(tool)  # type: ignore[arg-type]


def wrap_tool_langchain(tool: "Tool | LangchainTool") -> "LangchainTool":
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as langchain_tool

    if isinstance(tool, BaseTool):
        return tool

    return langchain_tool(tool)  # type: ignore[arg-type]


def wrap_tool_smolagents(tool: "Tool | SmolagentsTool") -> "SmolagentsTool":
    from smolagents import Tool as SmolagentsTool
    from smolagents import tool as smolagents_tool

    if isinstance(tool, SmolagentsTool):
        return tool

    # this wrapping needed until https://github.com/huggingface/smolagents/pull/1203 is merged and released
    @wraps(tool)  # type: ignore[arg-type]
    def wrapped_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        return tool(*args, **kwargs)  # type: ignore[operator]

    return smolagents_tool(wrapped_function)


def wrap_tool_llama_index(tool: "Tool | LlamaIndexTool") -> "LlamaIndexTool":
    from llama_index.core.tools import FunctionTool

    if isinstance(tool, FunctionTool):
        return tool

    return FunctionTool.from_defaults(tool)  # type: ignore[arg-type]


def wrap_tool_google(tool: "Tool | GoogleTool") -> "GoogleTool":
    from google.adk.tools import BaseTool, FunctionTool

    if isinstance(tool, BaseTool):
        return tool

    return FunctionTool(tool)  # type: ignore[arg-type]


def wrap_tool_agno(tool: Tool) -> Tool:
    # Agno lets you pass callables directly in as tools ❤️
    return tool


async def wrap_mcp_server(
    mcp_tool: MCPParams,
    agent_framework: AgentFramework,
) -> MCPServerBase:
    """Generic MCP server wrapper that can work with different frameworks
    based on the specified agent_framework
    """
    manager = get_mcp_server(mcp_tool, agent_framework)
    await manager.setup_tools()

    return manager


WRAPPERS: dict[AgentFramework, Callable[..., Any]] = {
    AgentFramework.GOOGLE: wrap_tool_google,
    AgentFramework.OPENAI: wrap_tool_openai,
    AgentFramework.LANGCHAIN: wrap_tool_langchain,
    AgentFramework.SMOLAGENTS: wrap_tool_smolagents,
    AgentFramework.LLAMA_INDEX: wrap_tool_llama_index,
    AgentFramework.AGNO: wrap_tool_agno,
}


def verify_callable(tool: Callable[..., Any]) -> None:
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
    tools: Sequence[Tool],
    agent_framework: AgentFramework,
) -> tuple[list[Tool], list[MCPServerBase]]:
    wrapper = WRAPPERS[agent_framework]

    wrapped_tools = list[Tool]()
    mcp_servers = list[MCPServerBase]()
    for tool in tools:
        # if it's MCPStdioParams or MCPSseParams, we need to wrap it in a server
        if isinstance(tool, MCPParams):
            # MCP adapters are usually implemented as context managers.
            # We wrap the server using `MCPServerBase` so the
            # tools can be used as any other callable.
            mcp_server = await wrap_mcp_server(tool, agent_framework)
            mcp_servers.append(mcp_server)
        elif callable(tool):
            verify_callable(tool)
            wrapped_tools.append(wrapper(tool))
        else:
            msg = f"Tool {tool} needs to be of type `MCPStdioParams`, `str` or `callable` but is {type(tool)}"
            raise ValueError(msg)

    return wrapped_tools, mcp_servers
