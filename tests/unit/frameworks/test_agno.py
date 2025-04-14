from unittest.mock import patch, MagicMock

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tools import (
    search_web,
    visit_webpage,
)


def test_load_agno_default():
    mock_agent = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.LiteLLM", mock_model),
    ):
        AnyAgent.create(AgentFramework.AGNO, AgentConfig(model_id="gpt-4o"))
        mock_agent.assert_called_once_with(
            name="any_agent",
            instructions="",
            model=mock_model(model="gpt-4o"),
            tools=[search_web, visit_webpage],
        )


def test_load_agno_agent_missing():
    with patch("any_agent.frameworks.agno.agno_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.AGNO, AgentConfig(model_id="gpt-4o"))
