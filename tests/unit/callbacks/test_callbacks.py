# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from any_agent import AgentConfig, AgentFramework, AgentRunError, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.testing.helpers import LLM_IMPORT_PATHS


class SampleCallback(Callback):
    def __init__(self) -> None:
        self.before_agent_invocation_called = False
        self.after_agent_invocation_called = False
        self.before_llm_called = False
        self.after_llm_called = False
        self.before_tool_called = False
        self.after_tool_called = False

    def after_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        self.after_agent_invocation_called = True
        return context

    def after_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        self.after_llm_called = True
        return context

    def after_tool_execution(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        self.after_tool_called = True
        return context

    def before_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        self.before_agent_invocation_called = True
        return context

    def before_tool_execution(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        self.before_tool_called = True
        return context

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        self.before_llm_called = True
        return context


class ExceptionCallback(SampleCallback):
    """Callback that throws an exception in before_llm_call."""

    def __init__(self, exception_message: str = "Callback exception") -> None:
        self.exception_message = exception_message
        super().__init__()

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        context = super().before_llm_call(context, *args, **kwargs)
        raise RuntimeError(self.exception_message)


def search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.
    """
    return "Information"


def create_agent(
    instructions: str,
    callbacks: list[Callback],
    tools: list[Callable[[str], str]] | None = None,
) -> AnyAgent:
    """Helper function to create an agent with common configuration."""
    config = AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions=instructions,
        callbacks=callbacks,
    )
    if tools:
        config.tools = tools  # type: ignore[assignment]

    return AnyAgent.create("tinyagent", config)


def run_agent_with_mock(
    agent: AnyAgent,
    prompt: str,
    mock_response: Any,
    expected_exception: type | None = None,
    exception_message: str | None = None,
) -> None:
    """Helper function to run agent with mocked response and optional exception handling."""
    import_path = LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]

    with patch(import_path, return_value=mock_response):
        if expected_exception:
            with pytest.raises(expected_exception, match=exception_message):
                agent.run(prompt)
        else:
            agent.run(prompt)


def test_callbacks(mock_any_llm_response: Any) -> None:
    callback = SampleCallback()
    agent = create_agent(
        instructions="Use the available tools to find information when needed",
        callbacks=[callback],
    )

    run_agent_with_mock(
        agent=agent,
        prompt="Hello!",
        mock_response=mock_any_llm_response,
    )

    # Verify that the callback methods were called
    assert callback.before_agent_invocation_called
    assert callback.after_agent_invocation_called
    assert callback.before_llm_called
    assert callback.after_llm_called
    assert callback.before_tool_called is False
    assert callback.after_tool_called is False


def test_tool_execution_callbacks(mock_any_llm_tool_call_response: Any) -> None:
    callback = SampleCallback()
    agent = create_agent(
        instructions="You must use the search_web tool to find information",
        callbacks=[callback],
        tools=[search_web],
    )

    run_agent_with_mock(
        agent=agent,
        prompt="Hello!",
        mock_response=mock_any_llm_tool_call_response,
    )

    # Verify that all callback methods were called
    assert callback.before_agent_invocation_called
    assert callback.after_agent_invocation_called
    assert callback.before_llm_called
    assert callback.after_llm_called
    assert callback.before_tool_called
    assert callback.after_tool_called


def test_callback_exception_causes_agent_exit(mock_litellm_response: Any) -> None:
    """Test that throwing an exception in a callback results in the agent exiting."""
    callback = ExceptionCallback("Test callback exception")
    agent = create_agent(
        instructions="Use the available tools to find information when needed",
        callbacks=[callback],
    )

    run_agent_with_mock(
        agent=agent,
        prompt="Search for information about the latest AI developments and summarize what you find",
        mock_response=mock_litellm_response,
        expected_exception=AgentRunError,
        exception_message="Test callback exception",
    )

    assert callback.before_agent_invocation_called
    assert callback.after_agent_invocation_called
    assert callback.before_llm_called
    assert callback.after_llm_called is False
    assert callback.before_tool_called is False
    assert callback.after_tool_called is False
