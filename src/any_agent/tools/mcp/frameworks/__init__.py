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


def _get_mcp_server(mcp_tool: MCPParams, agent_framework: AgentFramework) -> MCPServer:
    mapping = {
        AgentFramework.GOOGLE: GoogleMCPServer,
        AgentFramework.LANGCHAIN: LangchainMCPServer,
        AgentFramework.LLAMA_INDEX: LlamaIndexMCPServer,
        AgentFramework.OPENAI: OpenAIMCPServer,
        AgentFramework.AGNO: AgnoMCPServer,
        AgentFramework.SMOLAGENTS: SmolagentsMCPServer,
        AgentFramework.TINYAGENT: TinyAgentMCPServer,
    }
    cls = mapping.get(agent_framework)
    if cls is None:
        raise ValueError(f"Unsupported agent framework: {agent_framework}")
    return TypeAdapter[cls](cls).validate_python(
        {"mcp_tool": mcp_tool, "framework": agent_framework}
    )


__all__ = [
    "AgnoMCPServer",
    "GoogleMCPServer",
    "LangchainMCPServer",
    "LlamaIndexMCPServer",
    "MCPServer",
    "OpenAIMCPServer",
    "SmolagentsMCPServer",
    "TinyAgentMCPServer",
    "_get_mcp_server",
]
