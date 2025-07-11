import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.serving import A2AServingConfig
from any_agent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)
from any_agent.tools import a2a_tool_async
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.attributes import GenAI

from .conftest import (
    DATE_PROMPT,
    a2a_client_from_agent,
    assert_contains_current_date_info,
    get_datetime,
)


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    """Assert that agent_trace is valid and has final output."""
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_has_date_agent_tool_call(agent_trace: AgentTrace) -> None:
    """Assert that the agent trace contains a tool execution span for the date agent."""
    assert any(
        span.is_tool_execution()
        and span.attributes.get(GenAI.TOOL_NAME, None) == "call_date_agent"
        for span in agent_trace.spans
    )


@pytest.mark.asyncio
async def test_a2a_tool_async(agent_framework: AgentFramework) -> None:
    """Tests that an agent contacts another using A2A using the adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    skip_reason = {
        AgentFramework.SMOLAGENTS: "async a2a is not supported",
    }
    if agent_framework in skip_reason:
        pytest.skip(
            f"Framework {agent_framework}, reason: {skip_reason[agent_framework]}"
        )

    env_check = validate_environment(DEFAULT_SMALL_MODEL_ID)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    # Create date agent
    date_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        name="date_agent",
        model_id=DEFAULT_SMALL_MODEL_ID,
        description="Agent that can return the current date.",
        tools=[get_datetime],
        model_args=get_default_agent_model_args(agent_framework),
    )
    date_agent = await AnyAgent.create_async(
        agent_framework=agent_framework,
        agent_config=date_agent_cfg,
    )

    # Serve the agent and get client
    tool_agent_endpoint = "tool_agent"
    serving_config = A2AServingConfig(
        port=0,
        endpoint=f"/{tool_agent_endpoint}",
        log_level="info",
    )

    async with a2a_client_from_agent(date_agent, serving_config) as (_, server_url):
        # Create main agent with A2A tool
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            model_id=DEFAULT_SMALL_MODEL_ID,
            tools=[await a2a_tool_async(server_url, http_kwargs=DEFAULT_HTTP_KWARGS)],
            model_args=get_default_agent_model_args(agent_framework),
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
        )

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        assert_contains_current_date_info(str(agent_trace.final_output))
        _assert_has_date_agent_tool_call(agent_trace)
