import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig
from any_agent.tools import search_web, visit_webpage
from any_agent.tracing.trace import AgentTrace, is_tracing_supported


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent(
    agent_framework: AgentFramework,
    check_multi_tool_usage: Callable[[dict[str, Any]], None],
    tmp_path: Path,
) -> None:
    kwargs = {}

    if agent_framework is AgentFramework.TINYAGENT:
        pytest.skip(
            f"Skipping test for {agent_framework.name} because it does not support multi-agent"
        )

    agent_model = "gpt-4.1-nano"
    kwargs["model_id"] = agent_model
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(f"OPENAI_API_KEY needed for {agent_framework.name}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else None
    )
    main_agent = AgentConfig(
        instructions="Use the available tools to complete the task to obtain additional information to answer the query.",
        description="The orchestrator that can use other agents.",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )

    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id=agent_model,
            description="Agent that can search the web. It can find answers on the web if the query cannot be answered.",
            tools=[search_web],
            model_args=model_args,
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id=agent_model,
            description="Agent that can visit webpages",
            tools=[visit_webpage],
            model_args=model_args,
        ),
    ]
    traces = tmp_path / "traces"
    agent = AnyAgent.create(
        agent_framework=agent_framework,
        agent_config=main_agent,
        managed_agents=managed_agents,
        tracing=TracingConfig(
            output_dir=str(traces), console=False, save=True, cost_info=True
        ),
    )

    try:
        agent_trace = agent.run(
            "Which LLM agent framework is the most appropriate to execute SQL queries using grammar constrained decoding? I am working on a business environment with my own premises, and I would prefer hosting an open source model myself."
        )

        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output
        if is_tracing_supported(agent_framework):
            assert agent_trace.spans
            assert len(agent_trace.spans) > 0
            assert traces.exists()
            trace_files = [str(x) for x in traces.iterdir()]
            assert agent_trace.output_file in trace_files
            assert agent_framework.name in agent_trace.output_file
            cost_sum = agent_trace.get_total_cost()
            assert cost_sum.total_cost > 0
            assert cost_sum.total_cost < 1.00
            assert cost_sum.total_tokens > 0
            assert cost_sum.total_tokens < 20000
        if agent_framework not in (
            AgentFramework.AGNO,
            AgentFramework.GOOGLE,
            AgentFramework.TINYAGENT,
        ):
            assert traces.exists()
            log_files = traces.glob("*.json")
            with open(next(log_files)) as log_file:
                contents = json.load(log_file)
                check_multi_tool_usage(contents)
            assert agent_framework.name in str(next(traces.iterdir()).name)
    finally:
        agent.exit()
