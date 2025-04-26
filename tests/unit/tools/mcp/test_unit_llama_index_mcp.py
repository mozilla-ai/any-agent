from collections.abc import Sequence

import pytest
from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient

from any_agent.config import AgentFramework, MCPSseParams, Tool
from any_agent.tools import _get_mcp_server


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "llama_index_mcp_client",
    "llama_index_mcp_tool_spec",
)
async def test_llamaindex_mcp_sse_tools_loaded(
    mcp_sse_params_with_tools: MCPSseParams,
    tools: Sequence[Tool],
) -> None:
    server = _get_mcp_server(mcp_sse_params_with_tools, AgentFramework.LLAMA_INDEX)
    await server._setup_tools()

    assert server.tools == tools


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "llama_index_mcp_tool_spec",
)
async def test_llamaindex_mcp_sse_integration(
    mcp_sse_params_with_tools: MCPSseParams,
    llama_index_mcp_client: LlamaIndexMCPClient,
) -> None:
    server = _get_mcp_server(mcp_sse_params_with_tools, AgentFramework.LLAMA_INDEX)

    await server._setup_tools()

    llama_index_mcp_client.assert_called_once_with(
        command_or_url=mcp_sse_params_with_tools.url
    )
