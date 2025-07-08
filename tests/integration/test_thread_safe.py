import asyncio

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)


def mock_capital(query: str) -> str:
    """Perform a duckduckgo web search based on your query (think a Google search) then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    if "France" in query:
        return "The capital of France is Paris."
    if "Spain" in query:
        return "The capital of Spain is Madrid."
    return "No info"


@pytest.mark.asyncio
async def test_run_agent_concurrently(agent_framework: AgentFramework) -> None:
    """When an agent is run concurrently, state from the first run shouldn't bleed into the second run"""
    model_id = DEFAULT_SMALL_MODEL_ID
    env_check = validate_environment(model_id)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    agent = await AnyAgent.create_async(
        agent_framework,
        AgentConfig(
            model_id=model_id,
            instructions="You must use the tools to find an answer",
            model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
            tools=[mock_capital],
        ),
    )
    results = await asyncio.gather(
        agent.run_async("What is the capital of France?"),
        agent.run_async("What is the capital of Spain?"),
    )
    outputs = [r.final_output for r in results]
    assert all(o is not None for o in outputs)

    assert sum("Paris" in str(o) for o in outputs) == 1
    assert sum("Madrid" in str(o) for o in outputs) == 1

    first_spans = results[0].spans
    second_spans = results[1].spans
    assert second_spans[: len(first_spans)] != first_spans, (
        "Spans from the first run should not be in the second"
    )
