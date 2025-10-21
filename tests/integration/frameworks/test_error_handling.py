from typing import Any
from unittest.mock import patch

import pytest

from any_agent import (
    AgentConfig,
    AgentFramework,
    AgentRunError,
    AnyAgent,
)
from any_agent.callbacks import Callback, Context
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    LLM_IMPORT_PATHS,
    get_default_agent_model_args,
)
from any_agent.tracing.otel_types import StatusCode


class LimitLLMCalls(Callback):
    def __init__(self, max_llm_calls: int) -> None:
        self.max_llm_calls = max_llm_calls

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        if "n_llm_calls" not in context.shared:
            context.shared["n_llm_calls"] = 0

        context.shared["n_llm_calls"] += 1

        if context.shared["n_llm_calls"] > self.max_llm_calls:
            msg = "Reached limit of LLM Calls"
            raise RuntimeError(msg)

        return context


def test_runtime_error(
    agent_framework: AgentFramework,
) -> None:
    """An exception not caught by the framework should be caught by us.

    `AnyAgent.run_async` should catch and reraise an `AgentRunError`.

    The `AgentRunError.trace` should be retrieved.
    """
    kwargs = {}
    test_runtime_error_msg = "runtime error trap"

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID

    patch_function = LLM_IMPORT_PATHS.get(agent_framework)
    if not patch_function:
        err_msg = f"No patch function found for agent framework: {agent_framework}"
        raise ValueError(err_msg)

    with patch(patch_function) as llm_completion_path:
        llm_completion_path.side_effect = RuntimeError(test_runtime_error_msg)
        agent_config = AgentConfig(
            model_id=kwargs["model_id"],
            tools=[],
            model_args=get_default_agent_model_args(agent_framework),
        )
        agent = AnyAgent.create(agent_framework, agent_config)
        spans = []
        try:
            agent.run(
                "Write a four-line poem about agent frameworks.",
            )
        except AgentRunError as are:
            spans = are.trace.spans
            assert any(
                span.status.status_code == StatusCode.ERROR
                and span.status.description is not None
                and test_runtime_error_msg in span.status.description
                for span in spans
            )


def test_tool_error(
    agent_framework: AgentFramework,
) -> None:
    """An exception raised inside a tool will be caught by us.

    We make sure an appropriate Status is set to the tool execution span.
    We allow the Agent to try to recover from the tool calling failure.
    """
    exception_reason = "tool error trap"

    def search_web(query: str) -> str:
        """Perform a duckduckgo web search based on your query then returns the top search results.

        Args:
            query (str): The search query to perform.

        Returns:
            The top search results.

        """
        msg = exception_reason
        raise ValueError(msg)

    kwargs = {}

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID

    agent_config = AgentConfig(
        model_id=kwargs["model_id"],
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args=get_default_agent_model_args(agent_framework),
        callbacks=[LimitLLMCalls(max_llm_calls=5)],
    )

    agent = AnyAgent.create(agent_framework, agent_config)
    agent_trace = agent.run(
        "Check in the web which agent framework is the best. If the tool fails, don't try again, return final answer as failure.",
    )
    assert any(
        span.is_tool_execution()
        and span.status.status_code == StatusCode.ERROR
        and exception_reason in getattr(span.status, "description", "")
        for span in agent_trace.spans
    )
