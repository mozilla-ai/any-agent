import os

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig
from any_agent.tools import search_web, visit_webpage
from any_agent.tracing.trace import AgentTrace, _is_tracing_supported


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent(agent_framework: AgentFramework) -> None:
    kwargs = {}

    if agent_framework is AgentFramework.TINYAGENT:
        pytest.skip(
            f"Skipping test for {agent_framework.name} because it does not support multi-agent"
        )

    kwargs["model_id"] = "gpt-4.1-nano"
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else None
    )
    main_agent = AgentConfig(
        instructions="You must use the available agents to complete the task.",
        description="The orchestrator that can use other agents.",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )

    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id="gpt-4.1-nano",
            description="Agent that can search the web",
            tools=[search_web],
            model_args=model_args,
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id="gpt-4.1-nano",
            description="Agent that can visit webpages",
            tools=[visit_webpage],
            model_args=model_args,
        ),
    ]

    agent = AnyAgent.create(
        agent_framework=agent_framework,
        agent_config=main_agent,
        managed_agents=managed_agents,
        tracing=TracingConfig(console=False, cost_info=True),
    )
    agent_trace = agent.run("Which agent framework is the best?")

    assert agent_trace
    assert agent_trace.final_output
    if _is_tracing_supported(agent_framework):
        assert agent_trace.spans
        assert len(agent_trace.spans) > 0
        cost_sum = agent_trace.get_total_cost()
        assert cost_sum.total_cost > 0
        assert cost_sum.total_cost < 1.00
        assert cost_sum.total_tokens > 0
        assert cost_sum.total_tokens < 20000

    try:
        agent_trace = agent.run("Which agent framework is the best?")

        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output
        if _is_tracing_supported(agent_framework):
            assert agent_trace.spans
            assert len(agent_trace.spans) > 0
            cost_sum = agent_trace.get_total_cost()
            assert cost_sum.total_cost > 0
            assert cost_sum.total_cost < 1.00
            assert cost_sum.total_tokens > 0
            assert cost_sum.total_tokens < 20000
    finally:
        agent.exit()
