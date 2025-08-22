from unittest.mock import MagicMock

import pytest

from any_agent import AgentConfig, AgentFramework
from any_agent.config import MCPSse
from any_agent.tools import search_web
from any_agent.tools import _wrap_tools
from any_agent.tools.wrappers import WRAPPERS

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a")


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
    assert agent_card.capabilities.push_notifications
    assert not agent_card.capabilities.state_transition_history
    assert agent_card.url == "http://localhost:5000/"


@pytest.mark.asyncio
async def test_get_agent_card_with_mcp(  # type: ignore[no-untyped-def]
    agent_framework: AgentFramework, echo_sse_server
) -> None:
    # Skip SmolaAgents due to framework limitation with dynamically created functions
    if agent_framework == AgentFramework.SMOLAGENTS:
        pytest.skip(
            "SmolaAgents has issues with dynamically created MCP tool functions"
        )
    agent = MagicMock()
    agent.config = AgentConfig(model_id="foo", description="test agent")
    agent.framework = agent_framework

    # Use new MCP architecture
    mcp_config = MCPSse(url=echo_sse_server["url"])
    wrapped_tools, mcp_clients = await _wrap_tools([mcp_config], agent_framework)
    agent._tools = wrapped_tools

    agent_card = _get_agent_card(agent, A2AServingConfig())
    assert agent_card.name == "any_agent"
    assert agent_card.description == "test agent"
    assert len(agent_card.skills) == 3
    assert agent_card.skills[0].id == "any_agent-write_file"
    assert agent_card.skills[0].name == "write_file"
    # The MCP tool description now includes parameter information
    assert "Say hi back with the input text" in agent_card.skills[0].description


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
