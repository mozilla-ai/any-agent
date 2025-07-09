from uuid import uuid4

import pytest

# Import your agent and config
from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.serving import A2AServingConfig
from any_agent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    wait_for_server_async,
)

from .conftest import A2ATestHelpers, a2a_client_from_agent


@pytest.mark.asyncio
async def test_serve_async(test_port: int, a2a_test_helpers: A2ATestHelpers) -> None:
    # Create and serve the agent
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Directly answer the question without asking the user for input.",
            description="I'm an agent to help.",
            model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
        ),
    )

    # Use the context manager for proper cleanup
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
