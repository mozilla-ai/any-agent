import os

import pytest

from any_agent import AgentFramework, AgentSchema, AnyAgent


@pytest.mark.parametrize("framework", ("langchain", "openai", "smolagents"))
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Integration tests require `OPENAI_API_KEY` env var",
)
def test_load_and_run_agent(framework):
    agent_framework = AgentFramework(framework)
    agent_config = AgentSchema(model_id="gpt-4o-mini")
    agent = AnyAgent.create(agent_framework, agent_config)
    result = agent.run("What day is today?")
    assert result
