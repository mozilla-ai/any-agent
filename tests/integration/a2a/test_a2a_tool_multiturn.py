"""
This test is to test the task management functionality of the A2A classes.
Because the test isn't designed to test LLM performance, I mock the agent run_async method, this way
no calls are made to the LLM. I mock the responses to create a trace that mimics LLM output values,
and verify that the subsequent calls properly receive the previous conversation history.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest, TaskState
from pydantic import BaseModel

from any_agent import AgentConfig
from any_agent.config import AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent
from any_agent.serving import A2AServingConfig
from any_agent.serving.a2a.envelope import A2AEnvelope
from any_agent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    wait_for_server_async,
)
from any_agent.tools.a2a import a2a_tool_async
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace
from any_agent.tracing.attributes import GenAI
from any_agent.tracing.otel_types import (
    Resource,
    SpanContext,
    SpanKind,
    Status,
)

if TYPE_CHECKING:
    from typing import Any


class UserInfo(BaseModel):
    name: str
    job: str
    age: int | None = None


FIRST_TURN_PROMPT = "My name is Alice and I work as a software engineer."
FIRST_TURN_RESPONSE = "Hello, Alice! It's nice to meet you. You work as a software engineer. How old are you?"
SECOND_TURN_PROMPT = "What's my name, what do I do for work, and what's my age? Let me know if you need more information."
SECOND_TURN_RESPONSE = (
    "Your name is Alice, you work as a software engineer, and your age is 30."
)
THIRD_TURN_PROMPT = "My age is 30."
THIRD_TURN_RESPONSE = "Thank you for the information."


class MockConversationAgent(TinyAgent):
    """Mock agent implementation that simulates multi-turn conversation for testing."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.output_type = A2AEnvelope[UserInfo]
        self.turn_count = 0
        assert len(self.config.tools) == 1, (
            "This mock agent should have exactly one tool to be used for testing"
        )

    async def _load_agent(self) -> None:
        # Call parent's _load_agent to set up the basic structure
        await super()._load_agent()

    async def run_async(
        self, prompt: str, instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        # Verify that we don't have recursive "Previous conversation:" prefixes
        conversation_count = prompt.count("Previous conversation:")
        self._tools[0]()

        if self.turn_count == 0:
            # First turn: User introduces themselves
            assert prompt.count(FIRST_TURN_PROMPT) == 1, (
                f"First turn prompt should occur exactly once, but found {prompt.count(FIRST_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            # First turn should have no "Previous conversation:" prefix
            assert conversation_count == 0, (
                f"First turn should have no conversation history, but found {conversation_count} instances of 'Previous conversation:' in prompt: {prompt}"
            )
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=UserInfo(
                    name="Alice",
                    job="software engineer",
                    age=None,
                ),
            )
            return self._create_mock_trace(
                envelope, FIRST_TURN_RESPONSE, FIRST_TURN_PROMPT
            )
        if self.turn_count == 1:
            # Second turn: User asks for information back
            assert prompt.count(FIRST_TURN_PROMPT) == 1, (
                f"First turn prompt should occur exactly once, but found {prompt.count(FIRST_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            assert prompt.count(SECOND_TURN_PROMPT) == 1, (
                f"Second turn prompt should occur exactly once, but found {prompt.count(SECOND_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            # Second turn should have exactly 1 "Previous conversation:" prefix (not recursive)
            assert conversation_count == 1, (
                f"Second turn should have exactly 1 'Previous conversation:' prefix, but found {conversation_count} in prompt: {prompt}"
            )
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.input_required,
                data=UserInfo(
                    name="Alice",
                    job="software engineer",
                    age=None,
                ),
            )
            return self._create_mock_trace(envelope, SECOND_TURN_RESPONSE, prompt)
        if self.turn_count == 2:
            # Third turn: User provides age
            assert prompt.count(FIRST_TURN_PROMPT) == 1, (
                f"First turn prompt should occur exactly once, but found {prompt.count(FIRST_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            assert prompt.count(SECOND_TURN_PROMPT) == 1, (
                f"Second turn prompt should occur exactly once, but found {prompt.count(SECOND_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            assert prompt.count(THIRD_TURN_PROMPT) == 1, (
                f"Third turn prompt should occur exactly once, but found {prompt.count(THIRD_TURN_PROMPT)} occurrences in prompt: {prompt}"
            )
            # Third turn should have exactly 1 "Previous conversation:" prefix (not recursive)
            assert conversation_count == 1, (
                f"Third turn should have exactly 1 'Previous conversation:' prefix, but found {conversation_count} in prompt: {prompt}"
            )
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=UserInfo(
                    name="Alice",
                    job="software engineer",
                    age=30,
                ),
            )
            return self._create_mock_trace(envelope, THIRD_TURN_RESPONSE, prompt)
        msg = f"Unexpected turn count: {self.turn_count}"
        raise ValueError(msg)

    def _create_mock_trace(
        self, envelope: A2AEnvelope[UserInfo], agent_response: str, prompt: str
    ) -> AgentTrace:
        """Create a mock AgentTrace with minimal spans for testing."""

        spans = []
        spans.append(
            AgentSpan(
                name=f"call_llm {DEFAULT_SMALL_MODEL_ID}",
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
async def test_a2a_tool_multiturn() -> None:
    """Test that agents can maintain conversation context across multiple interactions."""
    call_count = 0

    def call_counter() -> None:
        """Callback to count the number of times the tool is called."""
        nonlocal call_count
        call_count += 1

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
        output_type=UserInfo,
        tools=[call_counter],
        model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
    )

    agent = MockConversationAgent(config)

    # Configure session management with short timeout for testing
    serving_config = A2AServingConfig(
        port=0,
        context_timeout_minutes=2,  # Short timeout for testing
    )

    server_handle = await agent.serve_async(serving_config=serving_config)
    server_url = f"http://localhost:{server_handle.port}"
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
                            "text": FIRST_TURN_PROMPT,
                        }
                    ],
                    "messageId": first_message_id,
                    "contextId": context_id,  # Link messages to same conversation
                },
            }

            request_1 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_1),  # type: ignore[arg-type]
            )
            response_1 = await client.send_message(
                request_1, http_kwargs=DEFAULT_HTTP_KWARGS
            )

            assert response_1 is not None
            # if the response is JSONRPCErrorResposne, log and raise an error
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}, Code: {response_1.root.error.code}, Data: {response_1.root.error.data}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_1.root.result.status.message.parts[0].root.text  # type: ignore[union-attr]
            )
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"

            send_message_payload_2 = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": SECOND_TURN_PROMPT,
                        }
                    ],
                    "messageId": str(uuid4()),
                    "contextId": response_1.root.result.contextId,  # Same context to continue conversation
                },
            }

            request_2 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_2),  # type: ignore[arg-type]
            )
            response_2 = await client.send_message(
                request_2, http_kwargs=DEFAULT_HTTP_KWARGS
            )

            assert response_2 is not None
            # if the response is JSONRPCErrorResposne, log and raise an error
            if hasattr(response_2.root, "error"):
                msg = f"Error: {response_2.root.error.message}, Code: {response_2.root.error.code}, Data: {response_2.root.error.data}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_2.root.result.status.message.parts[0].root.text  # type: ignore[union-attr]
            )
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"
            assert result.age is None
            assert response_2.root.result.status.state == TaskState.input_required  # type: ignore[union-attr]

            # Send a message to the agent to give the age
            send_message_payload_3 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": THIRD_TURN_PROMPT}],
                    "messageId": str(uuid4()),
                    "contextId": response_2.root.result.contextId,  # Same context to continue conversation
                    "taskId": response_2.root.result.id,  # type: ignore[union-attr]
                },
            }
            request_3 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_3),  # type: ignore[arg-type]
            )
            response_3 = await client.send_message(
                request_3, http_kwargs=DEFAULT_HTTP_KWARGS
            )
            assert response_3 is not None
            # if the response is JSONRPCErrorResposne, log and raise an error
            if hasattr(response_3.root, "error"):
                msg = f"Error: {response_3.root.error.message}, Code: {response_3.root.error.code}, Data: {response_3.root.error.data}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_3.root.result.status.message.parts[0].root.text  # type: ignore[union-attr]
            )
            assert response_3.root.result.status.state == TaskState.completed  # type: ignore[union-attr]
            assert result.age == 30

            assert call_count == 3

    finally:
        await server_handle.shutdown()


@pytest.mark.asyncio
async def test_a2a_tool_multiturn_async() -> None:
    """Test that agents can maintain conversation context across multiple interactions."""

    call_count = 0

    def call_counter() -> None:
        """Callback to count the number of times the tool is called."""
        nonlocal call_count
        call_count += 1

    # Create a mock agent that simulates multi-turn conversation
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,  # Using real model ID but will be mocked
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise."
            " If you need more information, ask the user for it."
        ),
        name="Structured UserInfo Agent",
        description="Agent with conversation memory for testing session management.",
        output_type=UserInfo,
        tools=[call_counter],
        model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
    )

    agent = MockConversationAgent(config)

    # Configure session management with short timeout for testing
    serving_config = A2AServingConfig(
        port=0,
        context_timeout_minutes=2,  # Short timeout for testing
    )

    server_handle = await agent.serve_async(serving_config=serving_config)
    server_url = f"http://localhost:{server_handle.port}"
    await wait_for_server_async(server_url)
    try:
        main_agent_cfg = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Use the available tools to obtain additional information to answer the query.",
            tools=[await a2a_tool_async(server_url)],
            model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=AgentFramework.TINYAGENT,
            agent_config=main_agent_cfg,
        )
        prompt = f"""
        Please talk to the structured UserInfo agent and interact with it. You'll contact it to ask three questions. Say the exact words from the prompt in your query to the agent.

        1. {FIRST_TURN_PROMPT}
        2. {SECOND_TURN_PROMPT}
        3. {THIRD_TURN_PROMPT}

        For question 1, when calling the tool, do not provide the context id or the task id.
        For question 2, when calling the tool, provide the context id but omit the task id.
        For question 3, when calling the tool, provide both the context id and the task id.
        """

        agent_trace = await main_agent.run_async(prompt)
        assert agent_trace.final_output is not None
        assert call_count == 3
    finally:
        await server_handle.shutdown()
