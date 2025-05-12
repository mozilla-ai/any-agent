from unittest.mock import MagicMock

import pytest

from any_agent import AgentConfig, AgentFramework
from any_agent.config import MCPSse, ServingConfig
from any_agent.tools import search_web
from any_agent.tools.mcp import _get_mcp_server
from any_agent.tools.wrappers import WRAPPERS

try:
    from any_agent.serving.agent_card import _get_agent_card
except ImportError:
    _get_agent_card = None  # type: ignore[assignment]


@pytest.mark.skipif(_get_agent_card is None, reason="a2a_samples is not installed")
def test_get_agent_card(agent_framework: AgentFramework) -> None:
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo")
    agent.framework = agent_framework
    agent._main_agent_tools = [WRAPPERS[agent_framework](search_web)]
    agent_card = _get_agent_card(agent, ServingConfig())
    assert agent_card.name == "any_agent"
    assert agent_card.description is None
    assert len(agent_card.skills) == 1
    assert agent_card.skills[0].id == "any_agent-search_web"
    assert agent_card.skills[0].name == "search_web"
    assert "Perform a duckduckgo web search" in agent_card.skills[0].description
    assert not agent_card.capabilities.streaming
    assert not agent_card.capabilities.pushNotifications
    assert not agent_card.capabilities.stateTransitionHistory
    assert agent_card.url == "http://localhost:5000/"


@pytest.mark.skipif(_get_agent_card is None, reason="a2a_samples is not installed")
@pytest.mark.asyncio
async def test_get_agent_card_with_mcp(  # type: ignore[no-untyped-def]
    agent_framework: AgentFramework, echo_sse_server
) -> None:
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo")
    agent.framework = agent_framework
    server = _get_mcp_server(MCPSse(url=echo_sse_server["url"]), agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        agent._main_agent_tools = list(server.tools[0].functions.values())  # type: ignore[union-attr]
    else:
        agent._main_agent_tools = server.tools

    agent_card = _get_agent_card(agent, ServingConfig())
    assert agent_card.name == "any_agent"
    assert agent_card.description is None
    assert len(agent_card.skills) == 3
    assert agent_card.skills[0].id == "any_agent-write_file"
    assert agent_card.skills[0].name == "write_file"
    assert agent_card.skills[0].description == "Say hi back with the input text"
