import os
import pytest
from unittest.mock import patch, MagicMock

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.frameworks.smolagents import (
    DEFAULT_AGENT_TYPE,
    DEFAULT_MODEL_CLASS,
)
from any_agent.tools import search_web, visit_webpage


def test_load_smolagent_default():
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(f"smolagents.{DEFAULT_AGENT_TYPE}", mock_agent),
        patch(f"smolagents.{DEFAULT_MODEL_CLASS}", mock_model),
        patch("smolagents.tool", mock_tool),
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
            tools=[mock_tool(search_web), mock_tool(visit_webpage)],
        )
        mock_model.assert_called_once_with(model_id="openai/o3-mini")


def test_load_smolagent_with_api_base_and_api_key_var():
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(f"smolagents.{DEFAULT_AGENT_TYPE}", mock_agent),
        patch(f"smolagents.{DEFAULT_MODEL_CLASS}", mock_model),
        patch("smolagents.tool", mock_tool),
        patch.dict(os.environ, {"OPENAI_API_KEY": "BAR"}),
    ):
        AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai/o3-mini",
                model_args=dict(
                    api_base="https://custom-api.example.com",
                    api_key_var="OPENAI_API_KEY",
                ),
            ),
        )

        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_model.return_value,
            managed_agents=[],
            verbosity_level=-1,
            tools=[mock_tool(search_web), mock_tool(visit_webpage)],
        )
        mock_model.assert_called_once_with(
            model_id="openai/o3-mini",
            api_base="https://custom-api.example.com",
            api_key="BAR",
        )


def test_load_smolagent_environment_error():
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(f"smolagents.{DEFAULT_AGENT_TYPE}", mock_agent),
        patch(f"smolagents.{DEFAULT_MODEL_CLASS}", mock_model),
        patch("smolagents.tool", mock_tool),
        patch.dict(os.environ, {}, clear=True),
    ):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            AnyAgent.create(
                AgentFramework.SMOLAGENTS,
                AgentConfig(
                    model_id="openai/o3-mini",
                    model_args=dict(api_key_var="MISSING_KEY"),
                ),
            )


def test_load_smolagents_agent_missing():
    with patch("any_agent.frameworks.smolagents.smolagents_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.SMOLAGENTS, AgentConfig(model_id="gpt-4o"))
