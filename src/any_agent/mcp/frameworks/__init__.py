from collections.abc import Sequence

from pydantic import TypeAdapter

from any_agent.config import AgentFramework, MCPParams

from .agno import AgnoMCPServer
from .google import GoogleMCPServer
from .langchain import LangchainMCPServer
from .llama_index import LlamaIndexMCPServer
from .openai import OpenAIMCPServer
from .smolagents import SmolagentsMCPServer
from .tinyagent import TinyAgentMCPServer

MCPServer = (
    AgnoMCPServer
    | GoogleMCPServer
    | LangchainMCPServer
    | LlamaIndexMCPServer
    | OpenAIMCPServer
    | SmolagentsMCPServer
    | TinyAgentMCPServer
)


def _wrap_mcp_server(mcp_tool: MCPParams, agent_framework: AgentFramework) -> MCPServer:
    return TypeAdapter[MCPServer](MCPServer).validate_python(
        {"mcp_tool": mcp_tool, "framework": agent_framework}
    )


async def wrap_mcp_servers(
    mcp_params: Sequence[MCPParams], agent_framework: AgentFramework
) -> list[MCPServer]:
    """Map the params from a MCP server to a MCPServer class."""
    mcp_servers_with_tools = list[MCPServer]()
    for mcp_param in mcp_params:
        mcp_server = _wrap_mcp_server(mcp_param, agent_framework)
        await mcp_server._setup_tools()
        mcp_servers_with_tools.append(mcp_server)
    return mcp_servers_with_tools


__all__ = [
    "AgnoMCPServer",
    "GoogleMCPServer",
    "LangchainMCPServer",
    "LlamaIndexMCPServer",
    "MCPServer",
    "OpenAIMCPServer",
    "SmolagentsMCPServer",
    "TinyAgentMCPServer",
    "_wrap_mcp_server",
    "wrap_mcp_servers",
]
