from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import (
    search_web,
    visit_webpage,
)


def test_load_agno_default() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.DEFAULT_MODEL_TYPE", mock_model),
    ):
        AnyAgent.create(AgentFramework.AGNO, AgentConfig(model_id="gpt-4o"))
        mock_agent.assert_called_once_with(
            name="any_agent",
            instructions=None,
            model=mock_model(model="gpt-4o"),
            tools=[],
        )


def test_load_agno_agent_missing() -> None:
    with patch("any_agent.frameworks.agno.agno_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.AGNO, AgentConfig(model_id="gpt-4o"))


def test_load_agno_multi_agent() -> None:
    mock_agent = MagicMock()
    mock_team = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.Team", mock_team),
        patch("any_agent.frameworks.agno.DEFAULT_MODEL_TYPE", mock_model),
    ):
        AnyAgent.create(
            AgentFramework.AGNO,
            AgentConfig(model_id="gpt-4o", tools=[search_web]),
            managed_agents=[
                AgentConfig(
                    model_id="gpt-4o-mini",
                    name="search-web-agent",
                    description="You can visit webpages",
                    tools=[
                        visit_webpage,
                    ],
                )
            ],
        )
        mock_agent.assert_called_once_with(
            name="search-web-agent",
            role="You can visit webpages",
            instructions=None,
            model=mock_model(model="gpt-4o-mini"),
            tools=[visit_webpage],
        )
        mock_team.assert_called_once_with(
            mode="collaborate",
            name="Team managed by agent any_agent",
            description=None,
            instructions=None,
            model=mock_model(model="gpt-4o"),
            members=[mock_agent.return_value],
            tools=[search_web],
        )


def test_run_agno_custom_args() -> None:
    mock_agent = MagicMock()
    # Create a mock response object with the required content attribute
    mock_response = MagicMock()
    mock_response.content = "mock response"

    # Set up the AsyncMock to return the mock response
    mock_agent_instance = AsyncMock()
    mock_agent_instance.arun.return_value = mock_response
    mock_agent.return_value = mock_agent_instance

    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.DEFAULT_MODEL_TYPE", mock_model),
    ):
        agent = AnyAgent.create(AgentFramework.AGNO, AgentConfig(model_id="gpt-4o"))
        result = agent.run("foo", retries=2)

        # Verify the result is as expected
        assert isinstance(result.final_output, str)
        assert result.final_output == "mock response"

        # Verify the agent was called with the right parameters
        mock_agent_instance.arun.assert_called_once_with("foo", retries=2)
