import multiprocessing
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient

# Import your agent and config
from any_agent import AgentConfig, AnyAgent
from any_agent.serving import A2AServingConfig
from tests.integration.helpers import DEFAULT_MODEL_ID, wait_for_server_async

from .conftest import A2ATestHelpers, a2a_client_from_agent


def serve_agent(port: int) -> None:
    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id=DEFAULT_MODEL_ID,
            instructions="Directly answer the question without asking the user for input.",
            description="I'm an agent to help.",
        ),
    )
    agent.serve(serving_config=A2AServingConfig(port=port))


@pytest.mark.asyncio
async def test_serve_sync(test_port: int, a2a_test_helpers: A2ATestHelpers) -> None:
    # Start the agent in a subprocess
    proc = multiprocessing.Process(target=serve_agent, args=(test_port,), daemon=True)
    proc.start()
    server_url = f"http://localhost:{test_port}"
    await wait_for_server_async(server_url)

    try:
        async with httpx.AsyncClient(timeout=10.0) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, server_url
            )

            request = a2a_test_helpers.create_send_message_request(
                text="What is an agent?"
            )
            response = await client.send_message(request)
            assert response is not None
    finally:
        proc.kill()
        proc.join()


@pytest.mark.asyncio
async def test_serve_async(test_port: int, a2a_test_helpers: A2ATestHelpers) -> None:
    # Create and serve the agent
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id=DEFAULT_MODEL_ID,
            instructions="Directly answer the question without asking the user for input.",
            description="I'm an agent to help.",
        ),
    )

    # Use the context manager for proper cleanup
    async with a2a_client_from_agent(agent, A2AServingConfig(port=test_port)) as (
        client,
        _,
    ):
        request = a2a_test_helpers.create_send_message_request(
            text="What is an agent?",
            message_id=uuid4().hex,
        )
        response = await client.send_message(request)
        assert response is not None
