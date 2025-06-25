"""
This test is to test the task management functionality of the A2A classes.
Because the test isn't designed to test LLM performance, I mock the agent run_async method, this way
no calls are made to the LLM. I mock the responses to create a trace that mimics LLM output values,
and verify that the subsequent calls properly receive the previous conversation history.
"""

from __future__ import annotations

import asyncio
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
from any_agent.serving.envelope import A2AEnvelope
from any_agent.tools.a2a import a2a_tool_async
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace
from any_agent.tracing.otel_types import (
    Resource,
    SpanContext,
    SpanKind,
    Status,
)
from tests.integration.helpers import wait_for_server_async

if TYPE_CHECKING:
    from typing import Any


class TestResult(BaseModel):
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
        self.output_type = A2AEnvelope[TestResult]
        self.turn_count = 0

    async def _load_agent(self) -> None:
        # Call parent's _load_agent to set up the basic structure
        await super()._load_agent()

    async def run_async(
        self, prompt: str, instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        if self.turn_count == 0:
            # First turn: User introduces themselves
            assert FIRST_TURN_PROMPT in prompt
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=TestResult(
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
            assert FIRST_TURN_PROMPT in prompt
            assert SECOND_TURN_PROMPT in prompt
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.input_required,
                data=TestResult(
                    name="Alice",
                    job="software engineer",
                    age=None,
                ),
            )
            return self._create_mock_trace(envelope, SECOND_TURN_RESPONSE, prompt)
        if self.turn_count == 2:
            # Third turn: User provides age
            assert FIRST_TURN_PROMPT in prompt
            assert SECOND_TURN_PROMPT in prompt
            assert THIRD_TURN_PROMPT in prompt
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=TestResult(
                    name="Alice",
                    job="software engineer",
                    age=30,
                ),
            )
            return self._create_mock_trace(envelope, THIRD_TURN_RESPONSE, prompt)
        msg = f"Unexpected turn count: {self.turn_count}"
        raise ValueError(msg)

    def _create_mock_trace(
        self, envelope: A2AEnvelope[TestResult], agent_response: str, prompt: str
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
                    "gen_ai.operation.name": "call_llm",
                    "gen_ai.request.model": "mock-model",
                    "gen_ai.input.messages": json.dumps(
                        [{"role": "user", "content": prompt}]
                    ),
                    "gen_ai.output": agent_response,
                    "gen_ai.output.type": "json",
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
        output_type=TestResult,
    )

    agent = MockConversationAgent(config)

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
                            "text": FIRST_TURN_PROMPT,
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
            result = TestResult.model_validate_json(
                response_1.root.result.status.message.parts[0].root.text
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
                    "taskId": response_1.root.result.id,
                },
            }

            request_2 = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload_2)
            )
            response_2 = await client.send_message(request_2)

            assert response_2 is not None
            result = TestResult.model_validate_json(
                response_2.root.result.status.message.parts[0].root.text
            )
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"
            assert result.age is None
            assert response_2.root.result.status.state == TaskState.input_required

            # Send a message to the agent to give the age
            send_message_payload_3 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": THIRD_TURN_PROMPT}],
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
            result = TestResult.model_validate_json(
                response_3.root.result.status.message.parts[0].root.text
            )
            assert response_3.root.result.status.state == TaskState.completed
            assert result.age == 30

    finally:
        await server.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_a2a_tool_multiturn_async() -> None:
    """Test that agents can maintain conversation context across multiple interactions."""

    # Create a mock agent that simulates multi-turn conversation
    config = AgentConfig(
        model_id="gpt-4o-mini",  # Using real model ID but will be mocked
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise."
            " If you need more information, ask the user for it."
        ),
        name="Structured TestResult Agent",
        description="Agent with conversation memory for testing session management.",
        output_type=TestResult,
    )

    agent = MockConversationAgent(config)

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

        class MainAgentAnswer(BaseModel):
            first_turn_success: bool
            second_turn_success: bool
            third_turn_success: bool

        main_agent_cfg = AgentConfig(
            model_id="gpt-4.1-nano",
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            tools=[await a2a_tool_async(server_url, http_kwargs={"timeout": 10.0})],
            output_type=MainAgentAnswer,
            model_args={
                "parallel_tool_calls": False  # to force it to talk to the agent one call at a time
            },
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=AgentFramework.TINYAGENT,
            agent_config=main_agent_cfg,
        )
        prompt = f"""
        Please talk to the structured testresult agent and interact with it. You'll contact it to ask three questions. Say the exact words from the prompt in your query to the agent.

        1. {FIRST_TURN_PROMPT}
        2. {SECOND_TURN_PROMPT}
        3. {THIRD_TURN_PROMPT}

        Make sure you appropriately continue the conversation by providing it with the task id if you want to continue the conversation.
        """

        agent_trace = await main_agent.run_async(prompt)
        assert agent_trace.final_output is not None
        assert isinstance(agent_trace.final_output, MainAgentAnswer)
        assert agent_trace.final_output.first_turn_success
        assert agent_trace.final_output.second_turn_success
        assert agent_trace.final_output.third_turn_success
    finally:
        await server.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
