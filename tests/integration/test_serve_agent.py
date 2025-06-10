import multiprocessing
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Import your agent and config
from any_agent import AgentConfig, AnyAgent
from any_agent.serving import A2AServingConfig

from .helpers import wait_for_server_async


def run_agent(port: int):
    agent = AnyAgent.create(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            description="I'm an agent to help with booking airbnbs",
        ),
    )
    agent.serve(serving_config=A2AServingConfig(port=port))


async def run_agent_async(port: int):
    agent = await AnyAgent.create_async(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            description="I'm an agent to help with booking airbnbs",
        ),
    )
    return await agent.serve_async(serving_config=A2AServingConfig(port=port))


@pytest.mark.asyncio
async def test_agent_serving_and_communication(test_port):
    """This test can be refactored to remove the need for multiproc, once we have support for control of the uvicorn server."""
    # Start the agent in a subprocess
    proc = multiprocessing.Process(target=run_agent, args=(test_port,), daemon=True)
    proc.start()
    server_url = f"http://localhost:{test_port}"
    await wait_for_server_async(server_url)

    try:
        async with httpx.AsyncClient() as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, server_url
            )
            send_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "how much is 10 USD in EUR?"}],
                    "messageId": str(uuid4()),
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )
            response = await client.send_message(request)
            assert response is not None
    finally:
        proc.kill()
        proc.join()


@pytest.mark.asyncio
async def test_agent_serving_and_communication_async(test_port):
    # Start the agent in a subprocess
    (task, server) = await run_agent_async(test_port)
    server_url = f"http://localhost:{test_port}"
    await wait_for_server_async(server_url)

    try:
        async with httpx.AsyncClient() as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, server_url
            )
            send_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "how much is 10 USD in EUR?"}],
                    "messageId": uuid4().hex,
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )
            response = await client.send_message(request)
            assert response is not None
    finally:
        await server.shutdown()
        task.cancel()
