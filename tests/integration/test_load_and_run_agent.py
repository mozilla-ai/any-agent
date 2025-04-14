import os

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tracing import setup_tracing

frameworks = [item for item in AgentFramework]


@pytest.mark.parametrize("framework", frameworks)
@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_agent(framework, tmp_path, refresh_tools):
    agent_framework = AgentFramework(framework)
    kwargs = {}

    if framework == "smolagents":
        kwargs["agent_type"] = "ToolCallingAgent"

    if framework != "openai":
        kwargs["model_id"] = "gemini/gemini-2.0-flash"
        if "GEMINI_API_KEY" not in os.environ:
            pytest.skip(f"GEMINI_API_KEY needed for {framework}")
    else:
        kwargs["model_id"] = "gpt-4o-mini"
        if "OPENAI_API_KEY" not in os.environ:
            pytest.skip(f"OPENAI_API_KEY needed for {framework}")
    # Agno not yet supported https://github.com/Arize-ai/openinference/issues/1302
    # Google ADK not yet supported https://github.com/Arize-ai/openinference/issues/1506
    if framework != "google" and framework != "agno":
        setup_tracing(agent_framework, str(tmp_path / "traces"))

    agent_config = AgentConfig(
        tools=["any_agent.tools.search_web"],
        instructions="Search the web to answer",
        **kwargs,
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    result = agent.run("Which agent framework is the best?")
    assert len(agent.tools) > 0
    assert result
