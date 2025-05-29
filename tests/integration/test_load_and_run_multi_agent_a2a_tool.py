import logging
import os

import pytest
from litellm.utils import validate_environment
from rich.logging import RichHandler

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import ServingConfig, TracingConfig
from any_agent.tools import a2a_tool
from any_agent.tracing.agent_trace import AgentTrace
from tests.conftest import build_tree

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
@pytest.mark.asyncio
async def test_load_and_run_multi_agent(
    agent_framework: AgentFramework,
) -> None:
    """Tests that an agent contacts another using A2A using the adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    if agent_framework in [
        AgentFramework.GOOGLE,
        AgentFramework.TINYAGENT,
        AgentFramework.SMOLAGENTS,
        AgentFramework.AGNO,
        AgentFramework.OPENAI,
        AgentFramework.LLAMA_INDEX,
    ]:
        pytest.skip(
            "https://github.com/mozilla-ai/any-agent/issues/357 tracks fixing so these tests can be re-enabled"
        )
    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-nano"
    agent_model = kwargs["model_id"]
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else None
    )

    main_agent = None
    served_agent = None
    served_task = None

    try:
        tool_agent_port = 5800
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
            tracing=TracingConfig(console=False, cost_info=True),
        )

        # SERVING PROPER

        served_agent = date_agent
        (served_task, served_server) = await served_agent.serve_async(
            serving_config=ServingConfig(
                port=tool_agent_port,
                endpoint=f"/{tool_agent_endpoint}",
                log_level="info",
            )
        )

        # Search agent is ready for card resolution

        logger.info(
            "Setting up agent",
            extra={
                "endpoint": f"http://localhost:{tool_agent_port}/{tool_agent_endpoint}"
            },
        )

        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            tools=[
                await a2a_tool(
                    f"http://localhost:{tool_agent_port}/{tool_agent_endpoint}"
                )
            ],
            model_args=model_args,
            **kwargs,  # type: ignore[arg-type]
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        agent_trace = await main_agent.run_async("What date and time is it right now?")

        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output

        logger.info("spans:")
        logger.info(build_tree(agent_trace.spans).model_dump_json(indent=2))

        logger.info(agent_trace.final_output)
        now = datetime.datetime.now()
        assert all(
            [
                str(now.year) in agent_trace.final_output,
                str(now.day) in agent_trace.final_output,
                now.strftime("%B") in agent_trace.final_output,
            ]
        )
        assert any(
            span.is_tool_execution()
            and span.attributes.get("gen_ai.tool.name", None) == "call_date_agent"
            for span in agent_trace.spans
        )

    finally:
        if served_server:
            await served_server.shutdown()
        if served_task:
            served_task.cancel()
