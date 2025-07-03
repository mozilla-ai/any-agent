# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from typing import Any
from unittest.mock import patch

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.callbacks import Callback, Context
from tests.unit.helpers import LITELLM_IMPORT_PATHS


class SampleCallback(Callback):
    def __init__(self):
        self.before_llm_called = False
        self.after_llm_called = False
        self.before_tool_called = False
        self.after_tool_called = False

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        self.after_llm_called = True
        return context

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        self.after_tool_called = True
        return context

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        self.before_tool_called = True
        return context

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        self.before_llm_called = True
        return context


def test_callbacks(
    mock_litellm_response: Any,
) -> None:
    callback = SampleCallback()
    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id="gpt-4o",
            instructions="Use the available tools to find information when needed",
            callbacks=[callback],
        ),
    )

    # Patch the appropriate litellm import path for this framework
    import_path = LITELLM_IMPORT_PATHS[AgentFramework.TINYAGENT]
    with patch(import_path, return_value=mock_litellm_response):
        # Run the agent with a prompt that should trigger tool usage
        agent.run(
            "Search for information about the latest AI developments and summarize what you find"
        )

        # Verify that the callback methods were called
        assert callback.before_llm_called
        assert callback.after_llm_called
        assert callback.before_tool_called is False
        assert callback.after_tool_called is False


def test_tool_execution_callbacks(
    mock_litellm_tool_call_response: Any,
) -> None:
    callback = SampleCallback()

    def search_web(query: str) -> str:
        """Perform a duckduckgo web search based on your query then returns the top search results.

        Args:
            query (str): The search query to perform.

        Returns:
            The top search results.

        """

        msg = "Information"
        return msg

    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id="gpt-4o",
            instructions="You must use the search_web tool to find information",
            tools=[search_web],
            callbacks=[callback],
        ),
    )
    # Patch the appropriate litellm import path for this framework
    import_path = LITELLM_IMPORT_PATHS[AgentFramework.TINYAGENT]
    with patch(import_path, return_value=mock_litellm_tool_call_response):
        # Run the agent
        agent.run("Search for information about the latest AI developments")

        # Verify that all callback methods were called
        assert callback.before_llm_called
        assert callback.after_llm_called
        assert callback.before_tool_called
        assert callback.after_tool_called
