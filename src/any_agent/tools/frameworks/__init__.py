from collections.abc import Iterable

from pydantic import TypeAdapter

from any_agent.config import AgentFramework, MCPParams, Tool
from any_agent.mcp.frameworks import MCPServer, _get_mcp_server

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


async def _wrap_mcp_server(
    tool: MCPParams, agent_framework: AgentFramework
) -> MCPServer:
    mcp_server = _get_mcp_server(tool, agent_framework)
    await mcp_server._setup_tools()
    return mcp_server


async def wrap_tools(
    tools: Iterable[Tool], agent_framework: AgentFramework
) -> tuple[list[AnyTool], list[MCPServer]]:
    """Wrap a list of tools for the specified framework."""
    mcp_servers = [
        await _wrap_mcp_server(tool, agent_framework)
        for tool in tools
        if isinstance(tool, MCPParams)
    ]

    wrapped_tools = [
        _wrap_tool(tool, agent_framework)
        for tool in tools
        if callable(tool) and not isinstance(tool, MCPParams)
    ]

    unsupported_tools = [
        tool for tool in tools if not isinstance(tool, MCPParams) and not callable(tool)
    ]

    if unsupported_tools:
        error_message = (
            f"Tool {unsupported_tools[0]} needs to be of type `MCPStdioParams`, "
            f"`str` or `callable` but is {type(unsupported_tools[0])}"
        )
        raise ValueError(error_message)

    return wrapped_tools, mcp_servers


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
