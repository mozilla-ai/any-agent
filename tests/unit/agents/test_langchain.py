from unittest.mock import patch, MagicMock

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tools import (
    search_web,
    visit_webpage,
)


def test_load_langchain_agent_default():
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()

    with (
        patch("any_agent.agents.langchain.create_react_agent", create_mock),
        patch("any_agent.agents.langchain.init_chat_model", model_mock),
        patch("langchain_core.tools.tool", tool_mock),
    ):
        AnyAgent.create(AgentFramework.LANGCHAIN, AgentConfig(model_id="gpt-4o"))
        model_mock.assert_called_once_with("gpt-4o")
        create_mock.assert_called_once_with(
            model=model_mock.return_value,
            tools=[tool_mock(search_web), tool_mock(visit_webpage)],
            prompt=None,
        )


def test_load_langchain_agent_missing():
    with patch("any_agent.agents.langchain.langchain_available", False):
        with pytest.raises(
            ImportError, match="You need to `pip install langchain langgraph`"
        ):
            AnyAgent.create(AgentFramework.LANGCHAIN, AgentConfig(model_id="gpt-4o"))
