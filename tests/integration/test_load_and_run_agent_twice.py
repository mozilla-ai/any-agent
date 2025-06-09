import asyncio
from typing import Any

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent

from .helpers import mock_search_web


@pytest.mark.asyncio
async def test_run_agent_twice(agent_framework: AgentFramework) -> None:
    """When an agent is run twice, state from the first run shouldn't bleed into the second run"""
    model_id = "gpt-4.1-nano"
    env_check = validate_environment(model_id)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else {}
    )
    model_args["temperature"] = 0.0

    agent = await AnyAgent.create_async(
        agent_framework,
        AgentConfig(
            model_id=model_id,
            instructions="You must use the tools to find an answer",
            model_args=model_args,
            tools=[mock_search_web],
        ),
    )
    results = await asyncio.gather(
        agent.run_async("What is the capital of France?"),
        agent.run_async("What is the capital of Spain?"),
    )
    outputs = [r.final_output for r in results]
    assert all(o is not None for o in outputs)

    assert sum("Paris" in o for o in outputs) == 1
    assert sum("Madrid" in o for o in outputs) == 1

    first_spans = results[0].spans
    second_spans = results[1].spans
    assert second_spans[: len(first_spans)] != first_spans, (
        "Spans from the first run should not be in the second"
    )
