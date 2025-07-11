import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest
import uvicorn
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TextPart,
)
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from any_agent import AgentConfig
from any_agent.config import AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent
from any_agent.serving import A2AServingConfig
from any_agent.serving.a2a.envelope import A2AEnvelope
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID, wait_for_server_async
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace
from any_agent.tracing.attributes import GenAI
from any_agent.tracing.otel_types import (
    Resource,
    SpanContext,
    SpanKind,
    Status,
)

from .conftest import DEFAULT_LONG_TIMEOUT, a2a_client_from_agent

FIRST_TURN_PROMPT = "What's the capital of Pennsylvania?"
FIRST_TURN_RESPONSE = "The capital of Pennsylvania is Harrisburg."


class StringInfo(BaseModel):
    value: str


class MockConversationAgent(TinyAgent):
    """Mock agent implementation that provides simple answers for testing."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.output_type = A2AEnvelope[StringInfo]

    async def _load_agent(self) -> None:
        # Call parent's _load_agent to set up the basic structure
        await super()._load_agent()

    async def run_async(
        self, prompt: str, instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        envelope = self.output_type(
            task_status=TaskState.input_required,
            data=StringInfo(value=FIRST_TURN_RESPONSE),
        )
        return self._create_mock_trace(envelope, FIRST_TURN_RESPONSE, FIRST_TURN_PROMPT)

    def _create_mock_trace(
        self, envelope: A2AEnvelope[StringInfo], agent_response: str, prompt: str
    ) -> AgentTrace:
        """Create a mock AgentTrace with minimal spans for testing."""

        spans = []
        spans.append(
            AgentSpan(
                name="call_llm gpt-4o-mini",
                kind=SpanKind.INTERNAL,
                status=Status(),
                context=SpanContext(span_id=123),
                attributes={
                    GenAI.OPERATION_NAME: "call_llm",
                    GenAI.REQUEST_MODEL: "mock-model",
                    GenAI.INPUT_MESSAGES: json.dumps(
                        [{"role": "user", "content": prompt}]
                    ),
                    GenAI.OUTPUT: agent_response,
                    GenAI.OUTPUT_TYPE: "json",
                },
                links=[],
                events=[],
                resource=Resource(),
            )
        )

        return AgentTrace(
            spans=spans,
            final_output=envelope,
        )

    @classmethod
    def create(cls, framework: AgentFramework | str, config: AgentConfig) -> AnyAgent:
        return cls(config)


@pytest.mark.asyncio
async def test_push_notification_non_streaming() -> None:
    """Test that the A2A server can send push notifications to a configured webhook.

    In non-streaming mode, the A2A server will send a single push notification at the end of message,
    which corresponds the the 'final' event in the TaskUpdater.

    """
    # Storage for notifications received by the webhook
    received_notifications = []

    async def webhook_handler(request: Request) -> JSONResponse:
        """Handle webhook notifications from the A2A server."""
        # Handle GET requests (for testing connectivity)
        if request.method == "GET":
            return JSONResponse({"status": "webhook is running"}, status_code=200)

        notification_data = await request.json()
        received_notifications.append(
            {"headers": dict(request.headers), "body": notification_data}
        )

        # Return success response
        return JSONResponse({"status": "received"}, status_code=200)

    # Create a mock agent that simulates multi-turn conversation
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,  # Using real model ID but will be mocked
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise."
            " If you need more information, ask the user for it."
        ),
        description="Agent with conversation memory for testing session management.",
    )

    agent = MockConversationAgent(config)

    # Configure session management with short timeout for testing
    serving_config = A2AServingConfig(
        port=0,
    )

    # Set up webhook server
    webhook_app = Starlette(
        routes=[Route("/webhook", webhook_handler, methods=["GET", "POST"])]
    )

    # Start webhook server on available port - bind to all interfaces for better accessibility
    webhook_config = uvicorn.Config(webhook_app, port=0)
    webhook_server = uvicorn.Server(webhook_config)
    webhook_task = asyncio.create_task(webhook_server.serve())

    # Wait for webhook server to start and get its port
    await asyncio.sleep(0.5)  # Give server more time to start
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

            # Configure push notifications in the initial message/send request
            # following the A2A specification example
            params = MessageSendParams(
                message=Message(
                    role=Role.user,
                    parts=[
                        Part(
                            root=TextPart(
                                kind="text",
                                text=FIRST_TURN_PROMPT,
                            )
                        )
                    ],
                    messageId=first_message_id,
                ),
                configuration=MessageSendConfiguration(
                    acceptedOutputModes=["text"],
                    pushNotificationConfig=PushNotificationConfig(url=webhook_url),
                ),
            )

            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}, Code: {response_1.root.error.code}, Data: {response_1.root.error.data}"
                raise RuntimeError(msg)
            if isinstance(response_1.root.result, Task):
                task_id = response_1.root.result.id
            else:  # Message
                task_id = response_1.root.result.messageId
            params.message.taskId = task_id

            # Send another message to the same task to trigger notifications
            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}, Code: {response_1.root.error.code}, Data: {response_1.root.error.data}"
                raise RuntimeError(msg)
            if isinstance(response_1.root.result, Task):
                assert response_1.root.result.id == task_id
            else:  # Message
                assert response_1.root.result.messageId == task_id

            response_2 = await client.send_message(request_1)
            if hasattr(response_2.root, "error"):
                msg = f"Error: {response_2.root.error.message}, Code: {response_2.root.error.code}, Data: {response_2.root.error.data}"
                raise RuntimeError(msg)
            if isinstance(response_2.root.result, Task):
                assert response_2.root.result.id == task_id
            else:  # Message
                assert response_2.root.result.messageId == task_id

            await asyncio.sleep(1)  # Give more time for notifications

            assert len(received_notifications) == 2

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
