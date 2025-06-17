import asyncio
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest, TaskState
from pydantic import BaseModel

from any_agent import AgentConfig, AnyAgent
from any_agent.serving import A2AServingConfig

from .helpers import wait_for_server_async


@pytest.mark.asyncio
async def test_task_management_multi_turn_conversation():
    """Test that agents can maintain conversation context across multiple interactions."""
    # Only running this test with the tinyagent framework because the goal here is to test A2A classes, not the agent framework.
    class TestResult(BaseModel):
        name: str
        job: str
        age: int | None = None

    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="gpt-4.1-nano",
            instructions=(
                "You are a helpful assistant that remembers our conversation. "
                "When asked about previous information, reference what was said earlier. "
                "Keep your responses concise."
                " If you need more information, ask the user for it."
            ),
            description="Agent with conversation memory for testing session management.",
            output_type=TestResult,
        ),
    )

    # Configure session management with short timeout for testing
    serving_config = A2AServingConfig(
        port=0,
        task_timeout_minutes=2,  # Short timeout for testing
    )

    (task, server) = await agent.serve_async(serving_config=serving_config)
    test_port = server.servers[0].sockets[0].getsockname()[1]
    server_url = f"http://localhost:{test_port}"
    await wait_for_server_async(server_url)

    try:
        async with httpx.AsyncClient(timeout=1500) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, server_url
            )

            # First interaction - establish context
            first_message_id = str(uuid4())
            context_id = str(uuid4())  # This will be our session identifier

            send_message_payload_1 = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "My name is Alice and I work as a software engineer.",
                        }
                    ],
                    "messageId": first_message_id,
                    "contextId": context_id,  # Link messages to same conversation
                },
            }

            request_1 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload_1)
            )
            response_1 = await client.send_message(request_1)

            assert response_1 is not None
            result = TestResult.model_validate_json(response_1.root.result.status.message.parts[0].root.text)
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"

            send_message_payload_2 = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "What's my name, what do I do for work, and what's my age? Let me know if you need more information.",
                        }
                    ],
                    "messageId": str(uuid4()),
                    "contextId": response_1.root.result.contextId,  # Same context to continue conversation
                    "taskId": response_1.root.result.id,
                },
            }

            request_2 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload_2)
            )
            response_2 = await client.send_message(request_2)

            assert response_2 is not None
            result = TestResult.model_validate_json(response_2.root.result.status.message.parts[0].root.text)
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"
            assert result.age == None
            assert response_2.root.result.status.state == TaskState.input_required

            # Send a message to the agent to give the age
            send_message_payload_3 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "My age is 30."}],
                    "messageId": str(uuid4()),
                    "contextId": response_1.root.result.contextId,  # Same context to continue conversation
                    "taskId": response_1.root.result.id,
                },
            }
            request_3 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload_3)
            )
            response_3 = await client.send_message(request_3)
            assert response_3 is not None
            result = TestResult.model_validate_json(response_3.root.result.status.message.parts[0].root.text)
            assert response_3.root.result.status.state == TaskState.completed
            assert result.age == 30

    finally:
        await server.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass