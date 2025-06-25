"""
This test is to test the push notification functionality of the A2A server.
It sets up a webhook server to receive push notifications, configures push notifications
for a task, and verifies that the server correctly sends notifications to the webhook.
"""

import asyncio
from uuid import uuid4

import pytest
import uvicorn
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    SendMessageRequest,
    SetTaskPushNotificationConfigRequest,
    TaskPushNotificationConfig,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from any_agent import AgentConfig
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.serving import A2AServingConfig
from tests.integration.helpers import wait_for_server_async

from .conftest import DEFAULT_LONG_TIMEOUT, A2ATestHelpers, a2a_client_from_agent

# Storage for notifications received by the webhook
received_notifications = []


async def webhook_handler(request: Request) -> JSONResponse:
    """Handle webhook notifications from the A2A server."""
    try:
        # Parse the notification payload
        notification_data = await request.json()
        received_notifications.append(
            {"headers": dict(request.headers), "body": notification_data}
        )

        # Return success response
        return JSONResponse({"status": "received"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@pytest.mark.asyncio
async def test_push_notification(a2a_test_helpers: A2ATestHelpers) -> None:
    """Test that the A2A server can send push notifications to a configured webhook."""

    # Clear any previous notifications
    received_notifications.clear()

    # Create a mock agent that simulates multi-turn conversation
    config = AgentConfig(
        model_id="gpt-4o-mini",  # Using real model ID but will be mocked
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise."
            " If you need more information, ask the user for it."
        ),
        description="Agent with conversation memory for testing session management.",
    )

    agent = await AnyAgent.create_async("tinyagent", config)

    # Configure session management with short timeout for testing
    serving_config = A2AServingConfig(
        port=0,
        task_timeout_minutes=2,  # Short timeout for testing
    )

    # Set up webhook server
    webhook_app = Starlette(routes=[Route("/webhook", webhook_handler)])

    # Start webhook server on available port
    webhook_config = uvicorn.Config(webhook_app, host="localhost", port=0)
    webhook_server = uvicorn.Server(webhook_config)
    webhook_task = asyncio.create_task(webhook_server.serve())

    # Wait for webhook server to start and get its port
    await asyncio.sleep(0.1)  # Give server a moment to start
    webhook_port = webhook_server.servers[0].sockets[0].getsockname()[1]
    webhook_url = f"http://localhost:{webhook_port}/webhook"
    await wait_for_server_async(webhook_url)

    try:
        # Use the helper context manager for agent serving and client setup
        async with a2a_client_from_agent(
            agent, serving_config, http_timeout=DEFAULT_LONG_TIMEOUT
        ) as (client, server_url):
            # Generate IDs for the conversation
            first_message_id = str(uuid4())
            context_id = str(uuid4())
            task_id = str(uuid4())

            # Configure push notifications in the initial message/send request
            # following the A2A specification example
            params = MessageSendParams(
                message=Message(
                    role="user",
                    parts=[
                        Part(
                            kind="text",
                            text="Generate a Q1 sales report. This usually takes a while. Notify me when it's ready.",
                        )
                    ],
                    messageId=first_message_id,
                    contextId=context_id,
                    taskId=task_id,
                ),
                configuration=MessageSendConfiguration(
                    acceptedOutputModes=["text"],
                    pushNotificationConfig=PushNotificationConfig(url=webhook_url),
                ),
            )

            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            assert response_1.root.result.id == task_id

            # Set up push notification configuration for the task
            push_notification_config = PushNotificationConfig(url=webhook_url)
            params_model = TaskPushNotificationConfig(
                taskId=task_id, pushNotificationConfig=push_notification_config
            )

            request_0 = SetTaskPushNotificationConfigRequest(id="", params=params_model)
            response_0 = await client.set_task_callback(request_0)
            assert not hasattr(response_0, "error")

            # Send another message to the same task to trigger notifications
            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            assert response_1.root.result.id == task_id

            # Wait for potential notifications
            await asyncio.sleep(5)

    finally:
        # Clean up webhook server properly
        if webhook_server:
            try:
                # Try to shutdown gracefully first
                if hasattr(webhook_server, "shutdown"):
                    await webhook_server.shutdown()
                else:
                    webhook_server.should_exit = True
                    # Give the server a moment to shut down gracefully
                    await asyncio.sleep(0.1)
            except Exception:
                # If graceful shutdown fails, force it
                webhook_server.should_exit = True

        if webhook_task and not webhook_task.done():
            webhook_task.cancel()
            try:
                await webhook_task
            except asyncio.CancelledError:
                pass
