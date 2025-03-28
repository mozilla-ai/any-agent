import os

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent


@pytest.mark.parametrize(
    "framework", ("langchain", "openai", "smolagents", "llama_index")
)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Integration tests require `OPENAI_API_KEY` env var",
)
def test_load_and_run_agent(framework):
    agent_framework = AgentFramework(framework)
    agent_config = AgentConfig(
        model_id="gpt-4o-mini", tools=["any_agent.tools.search_web"]
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    assert len(agent.tools) > 0
    result = agent.run("What day is today?")
    assert result
