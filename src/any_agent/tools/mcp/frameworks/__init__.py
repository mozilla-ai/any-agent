from .agno import AgnoMCPServer
from .google import GoogleMCPServer
from .langchain import LangchainMCPServer
from .llama_index import LlamaIndexMCPServer
from .openai import OpenAIMCPServer
from .smolagents import SmolagentsMCPServer

MCPServer = (
    AgnoMCPServer
    | GoogleMCPServer
    | LangchainMCPServer
    | LlamaIndexMCPServer
    | OpenAIMCPServer
    | SmolagentsMCPServer
)

__all__ = [
    "AgnoMCPServer",
    "GoogleMCPServer",
    "LangchainMCPServer",
    "LlamaIndexMCPServer",
    "MCPServer",
    "OpenAIMCPServer",
    "SmolagentsMCPServer",
]
