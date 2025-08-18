from uuid import uuid4

import pytest
import logging

# Import your agent and config
from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.logging import setup_logger
from any_agent.serving import A2AServingConfig
from any_agent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    wait_for_server_async,
)
from sse_starlette.sse import AppStatus

from .conftest import DATE_PROMPT, A2ATestHelpers, a2a_client_from_agent, get_datetime
from ..conftest import _get_deterministic_port  # noqa: TID252


@pytest.mark.asyncio
async def test_serve_async(
    request: pytest.FixtureRequest, a2a_test_helpers: A2ATestHelpers
) -> None:
    agent = await AnyAgent.create_async(
        AgentFramework.TINYAGENT,
        AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Directly answer the question without asking the user for input.",
            description="I'm an agent to help.",
            model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
        ),
    )

    test_port = _get_deterministic_port(
        request.node.name, AgentFramework.TINYAGENT.value
    )
    async with a2a_client_from_agent(agent, A2AServingConfig(port=test_port)) as (
        client,
        server_url,
    ):
        await wait_for_server_async(server_url)
        request = a2a_test_helpers.create_send_message_request(
            text="What is an agent?",
            message_id=uuid4().hex,
        )
        response = await client.send_message(request, http_kwargs=DEFAULT_HTTP_KWARGS)
        assert response is not None


@pytest.mark.asyncio
async def test_serve_streaming_async(
    request: pytest.FixtureRequest, a2a_test_helpers: A2ATestHelpers
) -> None:
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Use the available tools to obtain additional information to answer the query.",
            tools=[get_datetime],
            description="I'm an agent to help.",
            model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
        ),
    )

    test_port = _get_deterministic_port(
        request.node.name, AgentFramework.TINYAGENT.value
    )

    async with a2a_client_from_agent(
        agent, A2AServingConfig(port=test_port, stream_tool_usage=True)
    ) as (
        client,
        server_url,
    ):
        await wait_for_server_async(server_url)
        request = a2a_test_helpers.create_send_streaming_message_request(
            text=DATE_PROMPT,
            message_id=uuid4().hex,
        )
        responses = []
        async for response in client.send_message_streaming(
            request, http_kwargs=DEFAULT_HTTP_KWARGS
        ):
            responses.append(response)
            assert response is not None

        # 4 responses are for tool calls: 2 for get_datetime and 2 for final_answer
        assert len(responses) == 6
        AppStatus.get_or_create_exit_event().set()
