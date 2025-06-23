from unittest.mock import patch

from typing import AsyncGenerator, Any

import pytest
from litellm.types.utils import (
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    StreamingChoices,
)

from any_agent import (
    AgentConfig,
    AgentFramework,
    AgentRunError,
    AnyAgent,
)
from any_agent.tracing.otel_types import StatusCode


def search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    msg = "It's a trap!"
    raise ValueError(msg)


def test_tool_error_llm_mocked(
    agent_framework: AgentFramework,
) -> None:
    """An exception raised inside a tool will be caught by us.

    We make sure an appropriate Status is set to the tool execution span.
    We allow the Agent to try to recover from the tool calling failure.
    """

    skip_reason = {
        AgentFramework.AGNO: "Does not decide that the loop should end, and calls the tool the maximum amount of times before giving up",
        AgentFramework.GOOGLE: "When mocking the LLM tool call, the tool calling does not happen and the loop is ended",
    }
    if agent_framework in skip_reason:
        pytest.skip(
            f"Framework {agent_framework}, reason: {skip_reason[agent_framework]}"
        )

    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-nano"

    model_args = {"temperature": 0.0}

    agent_config = AgentConfig(
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args=model_args,
        **kwargs, # type: ignore[arg-type]
    )

    agent = AnyAgent.create(agent_framework, agent_config)

    x_function = Function(
        name="search_web", arguments={"query": "which agent framework is the best"}
    )

    x_tool_call = {"id": "call_12345xyz", "type": "function", "function": x_function}

    x_message = Message(tool_calls=[x_tool_call])

    fake_response = ModelResponse(choices=[Choices(message=x_message)])

    fake_chunk = ModelResponse(
        choices=[StreamingChoices(delta=Delta(tool_calls=[x_tool_call]))]
    )

    async def fake_iter() -> AsyncGenerator[ModelResponse]:
        yield fake_chunk

    def fake_resp(*args: list[Any], **kwargs: dict[str,Any]) -> ModelResponse:
        return fake_response

    patch_function = "litellm.acompletion"
    if agent_framework is AgentFramework.SMOLAGENTS:
        patch_function = "litellm.completion"
    if agent_framework is AgentFramework.GOOGLE:
        patch_function = "google.adk.models.lite_llm.acompletion"
    with (
        patch(patch_function) as acompletion_mock,
    ):
        if agent_framework in (
            AgentFramework.GOOGLE,
            AgentFramework.LLAMA_INDEX,
        ):
            acompletion_mock.return_value = fake_iter()
        else:
            acompletion_mock.side_effect = fake_resp

        agent_trace = None
        try:
            agent_trace = agent.run(
                "Check in the web which agent framework is the best.",
            )
        except AgentRunError as e:
            agent_trace = e.trace
        assert any(
            span.is_tool_execution()
            and span.status.status_code == StatusCode.ERROR
            and "It's a trap!" in getattr(span.status, "description", "")
            for span in agent_trace.spans
        )
