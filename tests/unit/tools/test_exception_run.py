from collections.abc import AsyncIterator, Iterator
from unittest.mock import patch

from litellm.types.utils import (
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from any_agent import (
    AgentConfig,
    AgentFramework,
    AnyAgent,
)
from any_agent.tracing.otel_types import StatusCode
from any_agent.testing.helpers import LLM_IMPORT_PATHS


class StreamingEndpoint:
    _responses: list[AsyncIterator[ModelResponseStream]]

    def __init__(self, responses: list[AsyncIterator[ModelResponseStream]]):
        self._responses = responses

    def next(self, **kwargs) -> AsyncIterator[ModelResponseStream]:  # type: ignore[no-untyped-def]
        return self._responses.pop(0)


def test_tool_error_llm_mocked(
    agent_framework: AgentFramework,
) -> None:
    """An exception raised inside a tool will be caught by us.

    We make sure an appropriate Status is set to the tool execution span.
    We allow the Agent to try to recover from the tool calling failure.
    """

    kwargs = {}

    kwargs["model_id"] = "mistral/mistral-small-latest"

    model_args = {"temperature": 0.0}

    exc_reason = "It's a trap!"

    def search_web(query: str) -> str:
        """Perform a duckduckgo web search based on your query then returns the top search results.

        Args:
            query (str): The search query to perform.

        Returns:
            The top search results.

        """
        msg = exc_reason
        raise ValueError(msg)

    agent_config = AgentConfig(
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )

    agent = AnyAgent.create(agent_framework, agent_config)

    give_up = """It appears that I am unable to perform web searches at the moment. However, I can provide some general information based on my existing
    knowledge.

    Some of the most popular and widely regarded agent frameworks include:

    • OpenAI's GPT-based frameworks: Used for creating conversational agents and AI assistants.
    • Rasa: An open-source framework for building conversational AI and chatbots.
    • Microsoft Bot Framework: A comprehensive framework for building enterprise-grade chatbots.
    • Dialogflow by Google: A natural language understanding platform for building conversational interfaces.
    • AgentScript: A framework for creating rule-based agents.

    If you need detailed comparisons or specific recommendations, I can help with that as well. Would you like me to do that?
    """

    fake_give_up_response = ModelResponse(
        choices=[Choices(message=Message(content=give_up))]
    )

    fake_give_up_chunk = ModelResponseStream(
        choices=[StreamingChoices(delta=Delta(content=give_up))]
    )

    fake_smolagents_final_answer_response = ModelResponse(
        choices=[
            Choices(
                message=Message(
                    tool_calls=[
                        {
                            "id": "call_67890abc",
                            "type": "function",
                            "function": Function(
                                name="final_answer", arguments={"answer": give_up}
                            ),
                        }
                    ]
                )
            )
        ]
    )

    function = Function(
        name="search_web", arguments={"query": "which agent framework is the best"}
    )

    tool_call = {"id": "call_12345xyz", "type": "function", "function": function}

    fake_tool_fail_response = ModelResponse(
        choices=[Choices(message=Message(tool_calls=[tool_call]))]
    )

    fake_tool_fail_chunk = ModelResponseStream(
        choices=[StreamingChoices(delta=Delta(tool_calls=[tool_call]))]
    )

    async def async_fake_tool_fail_chunk() -> AsyncIterator[ModelResponseStream]:
        yield fake_tool_fail_chunk

    async def async_fake_give_up_chunk() -> AsyncIterator[ModelResponseStream]:
        yield fake_give_up_chunk

    streaming = StreamingEndpoint(
        [async_fake_tool_fail_chunk(), async_fake_give_up_chunk()]
    )

    with (
        patch(LLM_IMPORT_PATHS[agent_framework]) as litellm_mock,
    ):
        if agent_framework in (AgentFramework.LLAMA_INDEX):
            litellm_mock.side_effect = streaming.next
        elif agent_framework in (AgentFramework.SMOLAGENTS):
            # For smolagents, we need to handle the ReAct pattern properly
            # First call should be the tool failure, then final_answer calls
            def smolagents_mock_generator() -> Iterator[ModelResponse]:
                yield fake_tool_fail_response
                # After tool failure, keep returning final_answer until agent terminates
                while True:
                    yield fake_smolagents_final_answer_response

            litellm_mock.side_effect = smolagents_mock_generator()
        else:
            # For other frameworks, just use the simple approach
            def other_mock_generator() -> Iterator[ModelResponse]:
                yield fake_tool_fail_response
                while True:
                    yield fake_give_up_response

            litellm_mock.side_effect = other_mock_generator()

        agent_trace = agent.run(
            "Check in the web which agent framework is the best.",
        )
        assert any(
            span.is_tool_execution()
            and span.status.status_code == StatusCode.ERROR
            and exc_reason in getattr(span.status, "description", "")
            for span in agent_trace.spans
        )
