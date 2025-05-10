from unittest.mock import MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import SmolagentsTool, search_web, visit_webpage
from any_agent.tools.frameworks import AnyTool


def test_load_smolagent_default() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE", mock_model),
    ):
        AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai/o3-mini",
            ),
        )

        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_model.return_value,
            managed_agents=[],
            verbosity_level=-1,
            tools=[SmolagentsTool(tool=search_web), SmolagentsTool(tool=visit_webpage)],
        )
        mock_model.assert_called_once_with(
            model_id="openai/o3-mini", api_base=None, api_key=None
        )


def test_load_smolagent_with_api_base() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE", mock_model),
    ):
        AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai/o3-mini",
                model_args={},
                api_base="https://custom-api.example.com",
            ),
        )

        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_model.return_value,
            managed_agents=[],
            verbosity_level=-1,
            tools=[SmolagentsTool(tool=search_web), SmolagentsTool(tool=visit_webpage)],
        )
        mock_model.assert_called_once_with(
            model_id="openai/o3-mini",
            api_base="https://custom-api.example.com",
            api_key=None,
        )


def test_load_smolagents_agent_missing() -> None:
    with patch("any_agent.frameworks.smolagents.smolagents_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.SMOLAGENTS, AgentConfig(model_id="gpt-4o"))


def test_run_smolagent_custom_args() -> None:
    mock_agent = MagicMock()
    mock_agent.return_value = MagicMock()
    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE"),
        patch("smolagents.tool"),
    ):
        agent = AnyAgent[AnyTool].create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai/o3-mini",
            ),
        )
        agent.run("foo", max_steps=30)
        mock_agent.return_value.run.assert_called_once_with("foo", max_steps=30)
