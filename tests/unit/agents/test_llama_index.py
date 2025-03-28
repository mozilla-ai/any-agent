from unittest.mock import patch, MagicMock

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tools import (
    search_web,
    visit_webpage,
)


def test_load_llama_index_agent_default():
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()
    from llama_index.core.tools import FunctionTool

    with (
        patch("any_agent.agents.llama_index.ReActAgent", create_mock),
        patch("llama_index.llms.openai.OpenAI", model_mock),
        patch.object(FunctionTool, "from_defaults", tool_mock),
    ):
        AnyAgent.create(AgentFramework.LLAMAINDEX, AgentConfig(model_id="gpt-4o"))
        model_mock.assert_called_once_with(model="gpt-4o")
        create_mock.assert_called_once_with(
            name="default-name",
            llm=model_mock.return_value,
            tools=[tool_mock(search_web), tool_mock(visit_webpage)],
        )


def test_load_llama_index_agent_missing():
    with patch("any_agent.agents.llama_index.llama_index_available", False):
        with pytest.raises(ImportError, match="You need to `pip install llama-index`"):
            AnyAgent.create(AgentFramework.LLAMAINDEX, AgentConfig(model_id="gpt-4o"))
