# pylint: disable=unused-argument, unused-variable
from collections.abc import Sequence
from typing import Any

import pytest

from any_agent.config import AgentFramework, MCPParams, MCPSse, Tool
from any_agent.tools import _get_mcp_server, _MCPConnection


@pytest.mark.asyncio
async def test_sse_tool_filtering(
    agent_framework: AgentFramework,
    sse_params_echo_server: MCPSse,
    tools: Sequence[str],
) -> None:
    reduced_tools = tools[:-1]
    new_params = sse_params_echo_server.model_copy(update={"tools": reduced_tools})
    server = _get_mcp_server(new_params, agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        # Check that only the specified tools are included
        assert set(server.tools[0].functions.keys()) == set(reduced_tools)  # type: ignore[union-attr]
    else:
        assert len(server.tools) == len(reduced_tools)  # ignore[arg-type]


@pytest.mark.asyncio
async def test_mcp_tools_loaded(
    agent_framework: AgentFramework,
    mcp_params: MCPParams,
    mcp_connection: _MCPConnection[Any],
    tools: Sequence[Tool],
) -> None:
    mcp_server = _get_mcp_server(mcp_params, agent_framework)

    await mcp_server._setup_tools(mcp_connection=mcp_connection)

    assert mcp_server.tools == list(tools)


def test_filter_tools(
    mcp_connection: _MCPConnection[Any], mcp_params: MCPParams
) -> None:
    """Test the _filter_tools method of _MCPConnection."""

    # Create tools with names
    class NamedTool:
        def __init__(self, name: str):
            self.name = name

    all_tools = [NamedTool("tool1"), NamedTool("tool2"), NamedTool("tool3")]

    # Case 1: No tools requested, all tools returned
    mcp_connection.mcp_tool = mcp_params.model_copy(update={"tools": None})
    filtered_tools = mcp_connection._filter_tools(all_tools)
    assert filtered_tools == all_tools

    # Case 2: Some tools requested, only those tools returned in the right order
    mcp_connection.mcp_tool = mcp_params.model_copy(
        update={"tools": ["tool3", "tool1"]}
    )
    filtered_tools = mcp_connection._filter_tools(all_tools)
    assert len(filtered_tools) == 2
    assert filtered_tools[0].name == "tool3"
    assert filtered_tools[1].name == "tool1"

    # Case 3: Request for non-existent tool raises ValueError
    mcp_connection.mcp_tool = mcp_params.model_copy(
        update={"tools": ["tool1", "nonexistent"]}
    )
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        mcp_connection._filter_tools(all_tools)
    assert "Missing: ['nonexistent']" in str(excinfo.value)
