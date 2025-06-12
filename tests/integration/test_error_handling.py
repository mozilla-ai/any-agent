import os
from unittest.mock import patch

import pytest
from litellm.utils import validate_environment

from any_agent import (
    AgentConfig,
    AgentFramework,
    AgentRunError,
    AnyAgent,
)
from any_agent.tracing.otel_types import StatusCode


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_runtime_error(
    agent_framework: AgentFramework,
) -> None:
    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-mini"
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = {"temperature": 0.0}

    exc_reason = "It's a trap!"

    patch_function = "litellm.acompletion"
    if agent_framework is AgentFramework.GOOGLE:
        patch_function = "google.adk.models.lite_llm.acompletion"
    elif agent_framework is AgentFramework.SMOLAGENTS:
        patch_function = "litellm.completion"

    with patch(patch_function) as litellm_path:
        litellm_path.side_effect = RuntimeError(exc_reason)
        agent_config = AgentConfig(
            tools=[],
            model_args=model_args,
            **kwargs,
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
                and exc_reason in span.status.description
                for span in spans
            )
