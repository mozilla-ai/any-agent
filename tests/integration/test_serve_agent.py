import asyncio
import multiprocessing
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Import your agent and config
from any_agent import AgentConfig, AnyAgent
from any_agent.config import ServingConfig

SERVER_PORT = 5000


def run_agent():
    agent = AnyAgent.create(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            description="I'm an agent to help with booking airbnbs",
        ),
    )
    agent.serve(serving_config=ServingConfig(port=SERVER_PORT))


async def run_agent_async():
    agent = await AnyAgent.create_async(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            description="I'm an agent to help with booking airbnbs",
        ),
    )
    return await agent.serve_async(serving_config=ServingConfig(port=SERVER_PORT))


@pytest.mark.asyncio
async def test_agent_serving_and_communication():
    """This test can be refactored to remove the need for multiproc, once we have support for control of the uvicorn server."""
    # Start the agent in a subprocess
    proc = multiprocessing.Process(target=run_agent, daemon=True)
    proc.start()
    await asyncio.sleep(5)

    try:
        async with httpx.AsyncClient() as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, f"http://localhost:{SERVER_PORT}"
            )
            send_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "how much is 10 USD in EUR?"}],
                    "messageId": uuid4().hex,
                },
            }
            request = SendMessageRequest(
                params=MessageSendParams(**send_message_payload)
            )
            response = await client.send_message(request)
            assert response is not None
    finally:
        proc.kill()
        proc.join()


@pytest.mark.asyncio
async def test_agent_serving_and_communication_async():
    # Start the agent in a subprocess
    (task, server) = await run_agent_async()
    try:
        async with httpx.AsyncClient() as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, f"http://localhost:{SERVER_PORT}"
            )
            send_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "how much is 10 USD in EUR?"}],
                    "messageId": uuid4().hex,
                },
            }
            request = SendMessageRequest(
                params=MessageSendParams(**send_message_payload)
            )
            response = await client.send_message(request)
            assert response is not None
    finally:
        await server.shutdown()
        task.cancel()
