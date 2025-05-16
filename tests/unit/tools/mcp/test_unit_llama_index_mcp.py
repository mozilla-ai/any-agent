from collections.abc import Generator, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.tools.mcp import BasicMCPClient as LlamaIndexMCPClient
from llama_index.tools.mcp import McpToolSpec as LlamaIndexMcpToolSpec

from any_agent.config import AgentFramework, MCPSse, MCPStdio, Tool
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


@pytest.mark.asyncio
async def test_llama_index_mcp_env() -> None:
    mcp_server = _get_mcp_server(
        MCPStdio(command="print('Hello MCP')", args=[], env={"FOO": "BAR"}),
        AgentFramework.LLAMA_INDEX,
    )
    mocked_class = MagicMock()
    mocked_class.return_value = AsyncMock()

    with (
        patch(
            "any_agent.tools.mcp.frameworks.llama_index.LlamaIndexMcpToolSpec",
            mocked_class,
        ),
    ):
        await mcp_server._setup_tools()
        assert mocked_class.call_args_list[0][1]["client"].env == {"FOO": "BAR"}
