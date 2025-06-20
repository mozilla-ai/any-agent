import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
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


def test_exception_trace(
    agent_framework: AgentFramework,
    tmp_path: Path,
) -> None:
    if agent_framework in (
        # Fails due to:
        # `('Recursion limit of 25 reached without hitting a stop condition. You can increase the ...oubleshooting/errors/GRAPH_RECURSION_LIMIT',)`
        AgentFramework.LANGCHAIN,
    ):
        pytest.skip(f"Framework {agent_framework.value} not currently passing the test")

    kwargs = {}

    tmp_file = "tmp.txt"
    # FIXME patch some call so an exception is triggered
    # An exception within a tool will be handled by the framework
    exc_reason = "the tool broke!"

    def fail_tool(text: str) -> None:
        """write the text to a file in the tmp_path directory

        Args:
            text (str): The text to write to the file.

        Returns:
            None
        """
        raise RuntimeError(exc_reason)

    kwargs["model_id"] = "gpt-4.1-mini"

    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else {}
    )
    model_args["temperature"] = 0.0
    tools = [
        fail_tool,
    ]
    agent_config = AgentConfig(
        tools=tools,  # type: ignore[arg-type]
        instructions="Answer the query.",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    agent = AnyAgent.create(agent_framework, agent_config)

    x_function = Function(
        name="fail_tool", arguments={"text":"test sample"}
    )

    x_tool_call = {
        "id": "call_12345xyz",
        "type": "function",
        "function": x_function
    }

    x_message = Message(
        tool_calls=[x_tool_call]
    )

    fake_response = ModelResponse(
        choices=[
            Choices(
                message=x_message
            )
        ]
    )

    fake_chunk = ModelResponse(
        choices=[
            StreamingChoices(
                delta=Delta(
                    tool_calls=[x_tool_call]
                )
            )
        ]
    )

    async def fake_iter():
        yield fake_chunk

    def fake_resp(*args, **kwargs):
        return fake_response

    patch_function = "litellm.acompletion"
    """
    if agent_framework is AgentFramework.GOOGLE:
        patch_function = "google.adk.models.lite_llm.acompletion"
    """
    if agent_framework is AgentFramework.SMOLAGENTS:
        patch_function = "litellm.completion"

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
        spans = []
        try:

            agent.run(
                "Write a four-line poem and use the tools to write it to a file.",
            )
        except AgentRunError as are:
            spans = are.trace.spans
        assert any(
            span.status.status_code == StatusCode.ERROR
            and exc_reason in span.status.description
            for span in spans
        )
