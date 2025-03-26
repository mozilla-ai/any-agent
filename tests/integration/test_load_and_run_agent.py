import os

import pytest

from any_agent import load_agent, run_agent, AgentSchema


@pytest.mark.parametrize("framework", ("langchain", "openai", "smolagents"))
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Integration tests require `OPENAI_API_KEY` env var",
)
def test_load_and_run_agent(framework):
    agent = load_agent("openai", AgentSchema(model_id="o3-mini"))
    result = run_agent(agent, "What day is today?")
    assert result
