from typing import assert_never

from any_agent.config import AgentFramework, MCPParams, MCPSse, MCPStdio

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


def _get_stdio_mcp_server(
    mcp_tool: MCPStdio, agent_framework: AgentFramework
) -> MCPServer:
    if agent_framework is AgentFramework.AGNO:
        from .agno import AgnoMCPServerStdio

        return AgnoMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.GOOGLE:
        from .google import GoogleMCPServerStdio

        return GoogleMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.LANGCHAIN:
        from .langchain import LangchainMCPServerStdio

        return LangchainMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.LLAMA_INDEX:
        from .llama_index import LlamaIndexMCPServerStdio

        return LlamaIndexMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.OPENAI:
        from .openai import OpenAIMCPServerStdio

        return OpenAIMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.SMOLAGENTS:
        from .smolagents import SmolagentsMCPServerStdio

        return SmolagentsMCPServerStdio(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.TINYAGENT:
        from .tinyagent import TinyAgentMCPServerStdio

        return TinyAgentMCPServerStdio(mcp_tool=mcp_tool)
    assert_never(agent_framework)


def _get_sse_mcp_server(mcp_tool: MCPSse, agent_framework: AgentFramework) -> MCPServer:
    if agent_framework is AgentFramework.AGNO:
        from .agno import AgnoMCPServerSse

        return AgnoMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.GOOGLE:
        from .google import GoogleMCPServerSse

        return GoogleMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.LANGCHAIN:
        from .langchain import LangchainMCPServerSse

        return LangchainMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.LLAMA_INDEX:
        from .llama_index import LlamaIndexMCPServerSse

        return LlamaIndexMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.OPENAI:
        from .openai import OpenAIMCPServerSse

        return OpenAIMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.SMOLAGENTS:
        from .smolagents import SmolagentsMCPServerSse

        return SmolagentsMCPServerSse(mcp_tool=mcp_tool)
    if agent_framework is AgentFramework.TINYAGENT:
        from .tinyagent import TinyAgentMCPServerSse

        return TinyAgentMCPServerSse(mcp_tool=mcp_tool)
    assert_never(agent_framework)


def _get_mcp_server(mcp_tool: MCPParams, agent_framework: AgentFramework) -> MCPServer:
    if isinstance(mcp_tool, MCPStdio):
        return _get_stdio_mcp_server(mcp_tool, agent_framework)
    if isinstance(mcp_tool, MCPSse):
        return _get_sse_mcp_server(mcp_tool, agent_framework)
    assert_never(mcp_tool)


__all__ = [
    "MCPServer",
    "_get_mcp_server",
]
