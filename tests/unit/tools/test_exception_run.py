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
    patched_function: str,
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
        with open(os.path.join(tmp_path, tmp_file), "w", encoding="utf-8") as f:
            f.write(text)

    kwargs["model_id"] = "gpt-4.1-mini"
    """
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")
    """

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

    fake_response = ModelResponse(
        choices=[
            Choices(
                message=Message(
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_12345xyz",
                            type="function",
                            function=Function(
                                name="fail_tool", arguments='{"text":"test sample"}'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    fake_chunk = ModelResponse(
        choices=[
            StreamingChoices(
                delta=Delta(
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_12345xyz",
                            type="function",
                            function=Function(
                                name="fail_tool", arguments='{"text":"test sample"}'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    async def fake_iter():
        yield fake_chunk

    with (
        patch(patched_function) as fw_agent_runtool,
        patch("litellm.acompletion") as acompletion_mock,
        patch("litellm.completion") as completion_mock,
    ):
        fw_agent_runtool.side_effect = RuntimeError(exc_reason)
        if agent_framework in (
            AgentFramework.GOOGLE,
            AgentFramework.LLAMA_INDEX,
        ):
            acompletion_mock.return_value = fake_iter()
            completion_mock.return_value = fake_iter()
        else:
            acompletion_mock.return_value = fake_response
            completion_mock.return_value = fake_response
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
