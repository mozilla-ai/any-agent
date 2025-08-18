import asyncio
import datetime
from asyncio import Task
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2AClient, A2ACardResolver
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

from any_agent import AnyAgent
from any_agent.serving import A2AServingConfig
from any_agent.testing.helpers import wait_for_server_async

if TYPE_CHECKING:
    from uvicorn.server import Server

# Constants
DEFAULT_TIMEOUT = 10.0
DEFAULT_LONG_TIMEOUT = 1500


async def get_client_from_agent_card_url(  # type: ignore[no-untyped-def]
    httpx_client: httpx.AsyncClient,
    base_url: str,
    agent_card_path: str = AGENT_CARD_WELL_KNOWN_PATH,
    http_kwargs: dict[str, Any] | None = None,
):
    agent_card: AgentCard = await A2ACardResolver(
        httpx_client, base_url=base_url, agent_card_path=agent_card_path
    ).get_agent_card(http_kwargs=http_kwargs)
    return A2AClient(httpx_client=httpx_client, agent_card=agent_card)


class A2ATestHelpers:
    """Helper methods for A2A testing."""

    @staticmethod
    def create_message_payload(
        text: str,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a standard A2A message payload."""
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": message_id or str(uuid4()),
            }
        }

        if context_id:
            payload["message"]["contextId"] = context_id
        if task_id:
            payload["message"]["taskId"] = task_id

        return payload

    @staticmethod
    def create_send_message_request(
        text: str,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> SendMessageRequest:
        """Create a SendMessageRequest with standard payload."""
        payload = A2ATestHelpers.create_message_payload(
            text=text,
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )

        return SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))

    @staticmethod
    def create_send_streaming_message_request(
        text: str,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> SendStreamingMessageRequest:
        """Create a SendStreamingMessageRequest with standard payload."""
        payload = A2ATestHelpers.create_message_payload(
            text=text,
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )

        return SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**payload)
        )


class A2AServedAgent:
    """Context manager for serving an agent and cleaning it up."""

    def __init__(
        self,
        agent: AnyAgent,
        serving_config: A2AServingConfig | None = None,
    ):
        self.agent = agent
        self.serving_config = serving_config or A2AServingConfig(port=0)
        self.task: Task[Any] | None = None
        self.server: Server | None = None
        self.server_url = ""

    async def __aenter__(self) -> "A2AServedAgent":
        """Start serving the agent."""
        server_handle = await self.agent.serve_async(serving_config=self.serving_config)

        self.task = server_handle.task
        self.server = server_handle.server

        # Get the actual port from the server
        assert self.server is not None
        endpoint = getattr(self.serving_config, "endpoint", "")
        self.server_url = f"http://localhost:{server_handle.port}{endpoint}"

        # Wait for server to be ready
        await wait_for_server_async(self.server_url)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the server."""
        if self.server:
            self.server.should_exit = True
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass


@asynccontextmanager
async def a2a_client_from_agent(
    agent: AnyAgent,
    serving_config: A2AServingConfig | None = None,
    http_timeout: float = DEFAULT_TIMEOUT,
) -> AsyncGenerator[tuple[A2AClient, str], None]:
    """Context manager that serves an agent and provides an A2A client for it."""
    async with A2AServedAgent(agent, serving_config) as served_agent:
        async with httpx.AsyncClient(timeout=http_timeout) as httpx_client:
            client = await get_client_from_agent_card_url(
                httpx_client, served_agent.server_url
            )
            yield client, served_agent.server_url


def get_datetime() -> str:
    """Return the current date and time - common test tool."""
    return str(datetime.datetime.now())


# Common date-related test data
DATE_PROMPT = (
    "What date and time is it right now? "
    "In your answer please include the year, month, day, and time. "
    "Example answer could be something like 'Today is December 15, 2024'"
)


def assert_contains_current_date_info(final_output: str) -> None:
    """Assert that the final output contains current date and time information."""
    now = datetime.datetime.now()
    assert all(
        [
            str(now.year) in final_output,
            str(now.day) in final_output,
            now.strftime("%B") in final_output,
        ]
    )


@pytest.fixture
def a2a_test_helpers() -> type[A2ATestHelpers]:
    """Fixture providing A2A test helper methods."""
    return A2ATestHelpers
