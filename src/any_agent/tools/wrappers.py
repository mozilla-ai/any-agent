import inspect
from collections.abc import Callable, MutableSequence, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from any_agent.config import AgentFramework, MCPParams, Tool
from any_agent.tools import (
    _get_mcp_server,
    _MCPServerBase,
)

if TYPE_CHECKING:
    from agents import Tool as AgentTool
    from google.adk.tools import BaseTool as GoogleTool
    from langchain_core.tools import BaseTool as LangchainTool
    from llama_index.core.tools import FunctionTool as LlamaIndexTool
    from smolagents import Tool as SmolagentsTool


def _wrap_tool_openai(tool: "Tool | AgentTool") -> "AgentTool":
    from agents import Tool as AgentTool
    from agents import function_tool

    if isinstance(tool, AgentTool):  # type: ignore[arg-type, misc]
        return tool  # type: ignore[return-value]

    return function_tool(tool)  # type: ignore[arg-type]


def _wrap_tool_langchain(tool: "Tool | LangchainTool") -> "LangchainTool":
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as langchain_tool

    if isinstance(tool, BaseTool):
        return tool

    return langchain_tool(tool)  # type: ignore[arg-type]


def _wrap_tool_smolagents(tool: "Tool | SmolagentsTool") -> "SmolagentsTool":
    from smolagents import Tool as SmolagentsTool
    from smolagents import tool as smolagents_tool

    if isinstance(tool, SmolagentsTool):
        return tool

    # this wrapping needed until https://github.com/huggingface/smolagents/pull/1203 is merged and released
    @wraps(tool)  # type: ignore[arg-type]
    def wrapped_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        return tool(*args, **kwargs)  # type: ignore[operator]

    return smolagents_tool(wrapped_function)


def _wrap_tool_llama_index(tool: "Tool | LlamaIndexTool") -> "LlamaIndexTool":
    from llama_index.core.tools import FunctionTool

    if isinstance(tool, FunctionTool):
        return tool

    return FunctionTool.from_defaults(tool)  # type: ignore[arg-type]


def _wrap_tool_google(tool: "Tool | GoogleTool") -> "GoogleTool":
    from google.adk.tools import BaseTool, FunctionTool

    if isinstance(tool, BaseTool):
        return tool

    return FunctionTool(tool)  # type: ignore[arg-type]


def _wrap_tool_agno(tool: Tool) -> Tool:
    # Agno lets you pass callables directly in as tools ❤️
    return tool


def _wrap_tool_tiny(tool: Tool) -> Tool:
    # Tiny lets you pass callables directly in as tools ❤️
    return tool


WRAPPERS: dict[AgentFramework, Callable[..., Any]] = {
    AgentFramework.GOOGLE: _wrap_tool_google,
    AgentFramework.OPENAI: _wrap_tool_openai,
    AgentFramework.LANGCHAIN: _wrap_tool_langchain,
    AgentFramework.SMOLAGENTS: _wrap_tool_smolagents,
    AgentFramework.LLAMA_INDEX: _wrap_tool_llama_index,
    AgentFramework.AGNO: _wrap_tool_agno,
    AgentFramework.TINYAGENT: _wrap_tool_tiny,
}


def verify_callable(tool: Callable[..., Any]) -> None:
    """Verify that `tool` is a valid callable.

    - It needs to have some sort of docstring that describes what it does
    - It needs to have typed argument
    - It needs to have a typed return.

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


T_co = TypeVar("T_co", covariant=True)


async def _wrap_tools(
    tools: Sequence[T_co],
    agent_framework: AgentFramework,
) -> tuple[list[T_co], list[_MCPServerBase[T_co]]]:
    wrapper = WRAPPERS[agent_framework]

    wrapped_tools = list[T_co]()
    mcp_servers: MutableSequence[_MCPServerBase[T_co]] = []
    for tool in tools:
        # if it's MCPStdio or MCPSse, we need to wrap it in a server
        if isinstance(tool, MCPParams):
            # MCP adapters are usually implemented as context managers.
            # We wrap the server using `MCPServerBase` so the
            # tools can be used as any other callable.
            mcp_server = _get_mcp_server(tool, agent_framework)
            await mcp_server._setup_tools()
            mcp_servers.append(mcp_server)  # type: ignore[arg-type]
        elif callable(tool):
            verify_callable(tool)
            wrapped_tools.append(wrapper(tool))
        else:
            msg = f"Tool {tool} needs to be of type `MCPStdio`, `str` or `callable` but is {type(tool)}"
            raise ValueError(msg)

    return wrapped_tools, mcp_servers  # type: ignore[return-value]
