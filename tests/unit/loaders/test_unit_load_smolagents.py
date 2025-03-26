import os
import pytest
from unittest.mock import patch, MagicMock


from any_agent.schema import AgentSchema
from any_agent.loaders.smolagents import (
    load_smolagents_agent,
    DEFAULT_AGENT_CLASS,
    DEFAULT_MODEL_CLASS,
)
from any_agent.tools import search_web, visit_webpage


def test_load_smolagent_default():
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(f"smolagents.{DEFAULT_AGENT_CLASS}", mock_agent),
        patch(f"smolagents.{DEFAULT_MODEL_CLASS}", mock_model),
        patch("smolagents.tool", mock_tool),
    ):
        load_smolagents_agent(
            AgentSchema(
                model_id="openai/o3-mini",
            ),
        )
        mock_agent.assert_called_once_with(
            name="default-name",
            model=mock_model.return_value,
            managed_agents=[],
            tools=[mock_tool(search_web), mock_tool(visit_webpage)],
        )
        mock_model.assert_called_once_with(model_id="openai/o3-mini")


def test_load_smolagent_with_api_base_and_api_key_var():
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch(f"smolagents.{DEFAULT_AGENT_CLASS}", mock_agent),
        patch(f"smolagents.{DEFAULT_MODEL_CLASS}", mock_model),
        patch("smolagents.tool", mock_tool),
        patch.dict(os.environ, {"OPENAI_API_KEY": "BAR"}),
    ):
        load_smolagents_agent(
            AgentSchema(
                model_id="openai/o3-mini",
                api_base="https://custom-api.example.com",
                api_key_var="OPENAI_API_KEY",
            ),
        )
        mock_agent.assert_called_once_with(
            name="default-name",
            model=mock_model.return_value,
            managed_agents=[],
            tools=[mock_tool(search_web), mock_tool(visit_webpage)],
        )
        mock_model.assert_called_once_with(
            model_id="openai/o3-mini",
            api_base="https://custom-api.example.com",
            api_key="BAR",
        )


def test_load_smolagent_environment_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            load_smolagents_agent(
                AgentSchema(model_id="openai/o3-mini", api_key_var="MISSING_KEY"),
            )
