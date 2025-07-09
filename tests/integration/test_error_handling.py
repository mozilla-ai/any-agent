from unittest.mock import patch

import pytest
from litellm.utils import validate_environment

from any_agent import (
    AgentConfig,
    AgentFramework,
    AgentRunError,
    AnyAgent,
)
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)
from any_agent.tracing.otel_types import StatusCode


def test_runtime_error(
    agent_framework: AgentFramework,
) -> None:
    """An exception not caught by the framework should be caught by us.

    `AnyAgent.run_async` should catch and reraise an `AgentRunError`.

    The `AgentRunError.trace` should be retrieved.
    """
    kwargs = {}

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    exc_reason = "It's a trap!"

    patch_function = "litellm.acompletion"
    if agent_framework is AgentFramework.GOOGLE:
        patch_function = "google.adk.models.lite_llm.acompletion"
    elif agent_framework is AgentFramework.SMOLAGENTS:
        patch_function = "litellm.completion"

    with patch(patch_function) as litellm_path:
        litellm_path.side_effect = RuntimeError(exc_reason)
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
                and exc_reason in span.status.description
                for span in spans
            )


def search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    msg = "It's a trap!"
    raise ValueError(msg)


def test_tool_error(
    agent_framework: AgentFramework,
) -> None:
    """An exception raised inside a tool will be caught by us.

    We make sure an appropriate Status is set to the tool execution span.
    We allow the Agent to try to recover from the tool calling failure.
    """
    kwargs = {}

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    agent_config = AgentConfig(
        model_id=kwargs["model_id"],
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args=get_default_agent_model_args(agent_framework),
    )

    agent = AnyAgent.create(agent_framework, agent_config)

    agent_trace = agent.run(
        "Check in the web which agent framework is the best.",
    )
    assert any(
        span.is_tool_execution()
        and span.status.status_code == StatusCode.ERROR
        and "It's a trap!" in getattr(span.status, "description", "")
        for span in agent_trace.spans
    )
