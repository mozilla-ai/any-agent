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
        patch("any_agent.frameworks.llama_index.ReActAgent", create_mock),
        patch("llama_index.llms.litellm.LiteLLM", model_mock),
        patch.object(FunctionTool, "from_defaults", tool_mock),
    ):
        AnyAgent.create(
            AgentFramework.LLAMAINDEX,
            AgentConfig(
                model_id="gemini/gemini-2.0-flash",
                instructions="You are a helpful assistant",
            ),
        )

        model_mock.assert_called_once_with(model="gemini/gemini-2.0-flash")
        create_mock.assert_called_once_with(
            name="any_agent",
            llm=model_mock.return_value,
            system_prompt="You are a helpful assistant",
            tools=[tool_mock(search_web), tool_mock(visit_webpage)],
        )


def test_load_llama_index_agent_missing():
    with patch("any_agent.frameworks.llama_index.llama_index_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.LLAMAINDEX, AgentConfig(model_id="gpt-4o"))


def test_load_langchain_multiagent():
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()
    from llama_index.core.tools import FunctionTool

    with (
        patch("any_agent.frameworks.llama_index.ReActAgent", create_mock),
        patch("any_agent.frameworks.llama_index.AgentWorkflow"),
        patch("llama_index.llms.litellm.LiteLLM", model_mock),
        patch.object(FunctionTool, "from_defaults", tool_mock),
    ):
        main_agent = AgentConfig(model_id="gpt-4.1-mini", description="Main agent")
        managed_agents = [
            AgentConfig(
                model_id="gpt-4.1-nano",
                tools=[
                    "any_agent.tools.search_web",
                    "any_agent.tools.visit_webpage",
                ],
                description="Managed agent",
            ),
        ]

        AnyAgent.create(
            AgentFramework.LLAMAINDEX, main_agent, managed_agents=managed_agents
        )

        create_mock.assert_any_call(
            name="managed_agent_0",
            description="Managed agent",
            llm=model_mock.return_value,
            tools=[
                tool_mock(search_web),
                tool_mock(visit_webpage),
            ],
            system_prompt=None,
            can_handoff_to=["any_agent"],
        )

        create_mock.assert_called_with(
            name="any_agent",
            description="Main agent",
            llm=model_mock.return_value,
            system_prompt=None,
            can_handoff_to=["managed_agent_0"],
            tools=[],
        )
