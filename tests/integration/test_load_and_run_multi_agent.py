import os

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig
from any_agent.tools import search_web, visit_webpage
from any_agent.tracing.agent_trace import AgentTrace


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent(
    agent_framework: AgentFramework,
) -> None:
    if agent_framework is AgentFramework.TINYAGENT:
        pytest.skip(
            "Skipping test for TINYAGENT because it does not support multi-agent"
        )
    if agent_framework is AgentFramework.LLAMA_INDEX:
        pytest.skip(
            "Skipping test for LLAMA_INDEX because it does not generate tool spans."
            "See https://github.com/run-llama/llama_index/issues/18776."
        )

    model_id = "gpt-4.1-nano"
    env_check = validate_environment("model_id")
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else None
    )
    main_agent = AgentConfig(
        instructions="Use the available tools to complete the task to obtain additional information to answer the query.",
        description="The orchestrator that can use other agents.",
        model_args=model_args,
        model_id=model_id,
    )

    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id=model_id,
            description="Agent that can search the web. It can find answers on the web if the query cannot be answered.",
            tools=[search_web],
            model_args=model_args,
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id=model_id,
            description="Agent that can visit webpages",
            tools=[visit_webpage],
            model_args=model_args,
        ),
    ]
    agent = AnyAgent.create(
        agent_framework=agent_framework,
        agent_config=main_agent,
        managed_agents=managed_agents,
        tracing=TracingConfig(console=False, cost_info=False),
    )

    try:
        agent_trace = agent.run(
            "Which LLM agent framework is the most appropriate to build muli-agent systems?"
        )

        assert isinstance(agent_trace, AgentTrace)

        tool_executions = [
            span for span in agent_trace.spans if span.is_tool_execution()
        ]
        assert tool_executions

    finally:
        agent.exit()
