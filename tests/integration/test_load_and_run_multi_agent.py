import os

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent


@pytest.mark.parametrize("framework", ("google", "openai", "smolagents"))
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Integration tests require `OPENAI_API_KEY` env var",
)
def test_load_and_run_multi_agent(framework):
    agent_framework = AgentFramework(framework)
    kwargs = {}
    if framework == "smolagents":
        kwargs["agent_type"] = "ToolCallingAgent"
    main_agent = AgentConfig(
        model_id="gpt-4o",
        instructions="Use the available agents to complete the task.",
        **kwargs,
    )
    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id="gpt-4o-mini",
            description="Agent that can search the web",
            tools=["any_agent.tools.search_web"],
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id="gpt-4o-mini",
            description="Agent that can visit webpages",
            tools=["any_agent.tools.visit_webpage"],
        ),
    ]
    if framework != "smolagents":
        managed_agents.append(
            AgentConfig(
                name="final_answer_agent",
                model_id="gpt-4o-mini",
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
