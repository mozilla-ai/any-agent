import os

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tracing import setup_tracing


@pytest.mark.parametrize(
    "framework", ("google", "openai", "langchain", "smolagents", "llama_index")
)
@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent(framework, tmp_path, refresh_tools):
    agent_framework = AgentFramework(framework)
    kwargs = {}
    if framework == "smolagents":
        kwargs["agent_type"] = "ToolCallingAgent"

    kwargs["model_id"] = "gpt-4.1-nano"
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(f"OPENAI_API_KEY needed for {framework}")

    if framework != "google":
        setup_tracing(agent_framework, str(tmp_path / "traces"))

    main_agent = AgentConfig(
        instructions="Use the available agents to complete the task.",
        description="The orchestrator that can use other agents.",
        model_args={"parallel_tool_calls": False},
        **kwargs,
    )
    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id="gpt-4.1-nano",
            description="Agent that can search the web",
            tools=["any_agent.tools.search_web"],
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id="gpt-4.1-nano",
            description="Agent that can visit webpages",
            tools=["any_agent.tools.visit_webpage"],
        ),
    ]
    if framework != "smolagents":
        managed_agents.append(
            AgentConfig(
                name="final_answer_agent",
                model_id="gpt-4.1-nano",
                description="Agent that can show the final answer",
                tools=["any_agent.tools.show_final_answer"],
                handoff=True,
            ),
        )
    agent = AnyAgent.create(
        agent_framework=agent_framework,
        agent_config=main_agent,
        managed_agents=managed_agents,
    )
    result = agent.run("Which agent framework is the best?")

    assert result
