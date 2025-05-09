from collections.abc import Generator, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

from any_agent.config import AgentFramework, MCPSse, Tool
from any_agent.tools import _get_mcp_server


@pytest.fixture
def llama_index_mcp_client() -> Generator[LlamaIndexMCPClient]:
    with patch(
        "any_agent.tools.mcp.frameworks.llama_index.LlamaIndexMCPClient"
    ) as mock_client:
        yield mock_client


@pytest.fixture
def llama_index_mcp_tool_spec(
    tools: Sequence[Tool],
) -> Generator[LlamaIndexMcpToolSpec]:
    tool_spec = MagicMock()
    tool_spec.to_tool_list_async = AsyncMock(return_value=tools)
    with patch(
        "any_agent.tools.mcp.frameworks.llama_index.LlamaIndexMcpToolSpec",
        return_value=tool_spec,
    ) as mock_tool_spec:
        yield mock_tool_spec


@pytest.mark.asyncio
@pytest.mark.usefixtures(
    "llama_index_mcp_tool_spec",
)
async def test_llamaindex_mcp_sse_integration(
    mcp_sse_params_with_tools: MCPSse,
    llama_index_mcp_client: LlamaIndexMCPClient,
) -> None:
    server = _get_mcp_server(mcp_sse_params_with_tools, AgentFramework.LLAMA_INDEX)

    await server._setup_tools()

    llama_index_mcp_client.assert_called_once_with(  # type: ignore[attr-defined]
        command_or_url=mcp_sse_params_with_tools.url
    )
