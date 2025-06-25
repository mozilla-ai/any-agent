import datetime

import pytest
from litellm.utils import validate_environment
from sse_starlette.sse import AppStatus

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import MCPSse
from any_agent.serving import MCPServingConfig
from any_agent.tracing.agent_trace import AgentTrace
from tests.integration.helpers import wait_for_server_async


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    """Assert that agent_trace is valid and has final output."""
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_contains_current_date_info(final_output: str) -> None:
    """Assert that the final output contains current date and time information."""
    now = datetime.datetime.now()
    assert all(
        [
            str(now.year) in final_output,
            str(now.day) in final_output,
            now.strftime("%B") in final_output,
        ]
    )


def _assert_has_tool_date_agent_call(agent_trace: AgentTrace) -> None:
    """Assert that the agent trace contains a tool execution span for the date agent."""
    assert any(
        span.is_tool_execution()
        and span.attributes.get("gen_ai.tool.name", None) == "as-tool-date_agent"
        for span in agent_trace.spans
    )


DATE_PROMPT = (
    "What date and time is it right now? "
    "In your answer please include the year, month, day, and time. "
    "Example answer could be something like 'Today is December 15, 2024'"
)


@pytest.mark.asyncio
async def test_mcp_serve(agent_framework: AgentFramework, test_port: int) -> None:
    """Tests that an agent contacts another as a tool using MCP.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    skip_reason = {
        AgentFramework.SMOLAGENTS: "async a2a is not supported",
    }
    if agent_framework in skip_reason:
        pytest.skip(
            f"Framework {agent_framework}, reason: {skip_reason[agent_framework]}"
        )
    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-nano"
    agent_model = kwargs["model_id"]
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = None

    main_agent = None
    served_task = None
    served_server = None

    try:
        tool_agent_endpoint = "tool_agent"

        # DATE AGENT

        import datetime

        def get_datetime() -> str:
            """Return the current date and time"""
            return str(datetime.datetime.now())

        date_agent_description = "Agent that can return the current date."
        date_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            name="date_agent",
            model_id=agent_model,
            description=date_agent_description,
            tools=[get_datetime],
            model_args=model_args,
        )
        date_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=date_agent_cfg,
        )

        # SERVING PROPER
        server_url = f"http://localhost:{test_port}/{tool_agent_endpoint}/sse"
        (served_task, served_server) = await date_agent.serve_async(
            serving_config=MCPServingConfig(
                port=test_port,
                endpoint=f"/{tool_agent_endpoint}",
                log_level="info",
            )
        )
        ping_url = f"http://localhost:{test_port}"
        # We cannot use the SSE stream, as it will not be closed in the request
        await wait_for_server_async(ping_url)

        # Search agent is ready for card resolution
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            tools=[
                MCPSse(url=server_url, client_session_timeout_seconds=300),
            ],
            model_args=model_args,
            **kwargs,  # type: ignore[arg-type]
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
        )

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        _assert_contains_current_date_info(str(agent_trace.final_output))
        _assert_has_tool_date_agent_call(agent_trace)

    finally:
        if main_agent:
            if mcp_conn := main_agent._mcp_servers[0].mcp_connection:
                await mcp_conn._exit_stack.aclose()
        if AppStatus.should_exit_event is not None:
            AppStatus.should_exit_event.set()
            AppStatus.should_exit_event = None
        if served_server:
            served_server.should_exit = True
        if served_task:
            await served_task
