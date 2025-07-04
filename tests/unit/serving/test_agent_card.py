from unittest.mock import MagicMock

import pytest

from any_agent import AgentConfig, AgentFramework
from any_agent.config import MCPSse
from any_agent.tools import search_web
from any_agent.tools.mcp import _get_mcp_server
from any_agent.tools.wrappers import WRAPPERS

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a.types")
pytest.importorskip("any_agent.serving.agent_card")
pytest.importorskip("typing.override")


from a2a.types import AgentSkill

from any_agent.serving import A2AServingConfig
from any_agent.serving.a2a.agent_card import _get_agent_card


def test_get_agent_card(agent_framework: AgentFramework) -> None:
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo", description="test agent")
    agent.framework = agent_framework
    agent._tools = [WRAPPERS[agent_framework](search_web)]
    agent_card = _get_agent_card(agent, A2AServingConfig())
    assert agent_card.name == "any_agent"
    assert agent_card.description == "test agent"
    assert len(agent_card.skills) == 1
    assert agent_card.skills[0].id == "any_agent-search_web"
    assert agent_card.skills[0].name == "search_web"
    assert "Perform a duckduckgo web search" in agent_card.skills[0].description
    assert not agent_card.capabilities.streaming
    assert not agent_card.capabilities.pushNotifications
    assert not agent_card.capabilities.stateTransitionHistory
    assert agent_card.url == "http://localhost:5000/"


@pytest.mark.asyncio
async def test_get_agent_card_with_mcp(  # type: ignore[no-untyped-def]
    agent_framework: AgentFramework, echo_sse_server
) -> None:
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo", description="test agent")
    agent.framework = agent_framework
    server = _get_mcp_server(MCPSse(url=echo_sse_server["url"]), agent_framework)
    await server._setup_tools()
    if agent_framework is AgentFramework.AGNO:
        agent._tools = list(server.tools[0].functions.values())  # type: ignore[union-attr]
    else:
        agent._tools = server.tools

    agent_card = _get_agent_card(agent, A2AServingConfig())
    assert agent_card.name == "any_agent"
    assert agent_card.description == "test agent"
    assert len(agent_card.skills) == 3
    assert agent_card.skills[0].id == "any_agent-write_file"
    assert agent_card.skills[0].name == "write_file"
    assert agent_card.skills[0].description == "Say hi back with the input text"


def test_get_agent_card_with_explicit_skills(agent_framework: AgentFramework) -> None:
    """Test that when skills are explicitly provided in A2AServingConfig, they are used instead of inferring from tools."""
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo", description="test agent")
    agent.framework = agent_framework
    # Give the agent some tools that would normally be used to infer skills
    agent._tools = [WRAPPERS[agent_framework](search_web)]

    # Create explicit skills that are different from what would be inferred
    explicit_skills = [
        AgentSkill(
            id="custom-skill-1",
            name="custom_function_1",
            description="This is a custom skill that does something amazing",
            tags=["custom", "test"],
        ),
        AgentSkill(
            id="custom-skill-2",
            name="custom_function_2",
            description="Another custom skill for testing purposes",
            tags=["custom", "demo"],
        ),
    ]

    serving_config = A2AServingConfig(skills=explicit_skills)
    agent_card = _get_agent_card(agent, serving_config)

    # Verify basic agent card properties
    assert agent_card.name == "any_agent"
    assert agent_card.description == "test agent"

    # Verify that the explicit skills are used (not inferred from tools)
    assert len(agent_card.skills) == 2
    assert agent_card.skills[0].id == "custom-skill-1"
    assert agent_card.skills[1].id == "custom-skill-2"

    # Verify that the skills are NOT the ones that would be inferred from search_web tool
    skill_names = [skill.name for skill in agent_card.skills]
    assert (
        "search_web" not in skill_names
    )  # This would be present if skills were inferred from tools
