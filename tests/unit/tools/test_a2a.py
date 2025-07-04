import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a.types")
pytest.importorskip("any_agent.tools.a2a")

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    Part,
    Role,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from any_agent.tools.a2a import a2a_tool, a2a_tool_async


# Helper functions and fixtures
def mock_agent_card(name: str = "test_agent") -> AgentCard:
    """Fixture providing a mock AgentCard for testing."""
    return AgentCard(
        capabilities=AgentCapabilities(),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        description="Test agent",
        name=name,
        skills=[],
        url="http://example.com/test",
        version="1.0.0",
    )


@asynccontextmanager
async def mock_a2a_tool(
    agent_card: AgentCard, response: Any
) -> AsyncGenerator[tuple[Any, AsyncMock], None]:
    """Context manager that sets up A2A mocks and returns the created tool."""
    with (
        patch("any_agent.tools.a2a.A2ACardResolver") as mock_resolver,
        patch("any_agent.tools.a2a.A2AClient") as mock_client_class,
    ):
        # Setup resolver mock
        mock_resolver_instance = AsyncMock()
        mock_resolver_instance.get_agent_card.return_value = agent_card
        mock_resolver.return_value = mock_resolver_instance

        # Setup client mock
        mock_client_instance = AsyncMock()
        mock_client_instance.send_message.return_value = SendMessageResponse(
            root=response
        )
        mock_client_class.return_value = mock_client_instance

        # Create and yield the tool
        tool = await a2a_tool_async("http://example.com/test")
        yield tool, mock_client_instance


def create_task_response() -> Task:
    """Factory function to create a Task response."""
    return Task(
        id="task-123",
        contextId="context-456",
        kind="task",
        status=TaskStatus(
            state=TaskState.completed,
            message=Message(
                role=Role.agent,
                parts=[Part(root=TextPart(text="Task completed successfully"))],
                messageId="msg-789",
                taskId="task-123",
            ),
            timestamp="2024-01-01T12:00:00Z",
        ),
    )


def create_error_response() -> JSONRPCErrorResponse:
    """Factory function to create an error response."""
    return JSONRPCErrorResponse(
        id="req-789",
        jsonrpc="2.0",
        error=JSONRPCError(
            code=-32601,
            message="Method not found",
            data={"details": "The requested method is not available"},
        ),
    )


def test_async_tool_name_default() -> None:
    """Test that async tool uses agent card name by default."""
    fun_name = "some_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = asyncio.run(a2a_tool_async("http://example.com/test"))
        assert created_fun.__name__ == f"call_{fun_name}"


def test_async_tool_name_specific() -> None:
    """Test that async tool accepts custom name parameter."""
    other_name = "other_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card("some_name")
        created_fun = asyncio.run(a2a_tool_async("http://example.com/test", other_name))
        assert created_fun.__name__ == f"call_{other_name}"


def test_async_tool_name_whitespace_handling() -> None:
    """Test that async tool properly handles whitespace in names."""
    fun_name = "  some_n  ame  "
    corrected_fun_name = "some_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = asyncio.run(a2a_tool_async("http://example.com/test"))
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_async_tool_name_exotic_whitespace() -> None:
    """Test that async tool handles various whitespace characters."""
    fun_name = " \n so \t me_n\t ame  \n"
    corrected_fun_name = "so_me_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = asyncio.run(a2a_tool_async("http://example.com/test"))
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_async_tool_name_specific_whitespace() -> None:
    """Test that async tool handles whitespace in custom names."""
    other_name = " \n oth \t er_n\t ame  \n"
    corrected_other_name = "oth_er_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card("some_name")
        created_fun = asyncio.run(a2a_tool_async("http://example.com/test", other_name))
        assert created_fun.__name__ == f"call_{corrected_other_name}"


def test_sync_tool_name_default() -> None:
    """Test that sync tool uses agent card name by default."""
    fun_name = "some_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = a2a_tool("http://example.com/test")
        assert created_fun.__name__ == f"call_{fun_name}"


def test_sync_tool_name_specific() -> None:
    """Test that sync tool accepts custom name parameter."""
    other_name = "other_name"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card("some_name")
        created_fun = a2a_tool("http://example.com/test", other_name)
        assert created_fun.__name__ == f"call_{other_name}"


def test_sync_tool_name_whitespace_handling() -> None:
    """Test that sync tool properly handles whitespace in names."""
    fun_name = "  some_n  ame  "
    corrected_fun_name = "some_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = a2a_tool("http://example.com/test")
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_sync_tool_name_exotic_whitespace() -> None:
    """Test that sync tool handles various whitespace characters."""
    fun_name = " \n so \t me_n\t ame  \n"
    corrected_fun_name = "so_me_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card(fun_name)
        created_fun = a2a_tool("http://example.com/test")
        assert created_fun.__name__ == f"call_{corrected_fun_name}"


def test_sync_tool_name_specific_whitespace() -> None:
    """Test that sync tool handles whitespace in custom names."""
    other_name = " \n oth \t er_n\t ame  \n"
    corrected_other_name = "oth_er_n_ame"
    with patch("any_agent.tools.a2a.A2ACardResolver.get_agent_card") as agent_card_mock:
        agent_card_mock.return_value = mock_agent_card("some_name")
        created_fun = a2a_tool("http://example.com/test", other_name)
        assert created_fun.__name__ == f"call_{corrected_other_name}"


@pytest.mark.asyncio
async def test_handles_task_response() -> None:
    """Test that the a2a_tool properly handles receiving a Task message back from the server."""
    task_response = create_task_response()
    success_response = SendMessageSuccessResponse(
        id="req-123", jsonrpc="2.0", result=task_response
    )

    async with mock_a2a_tool(mock_agent_card(), success_response) as (
        tool,
        mock_client,
    ):
        result = await tool("Test query", None, None)

        # Verify the result is the expected dictionary format
        if task_response.status.message:
            expected_result = {
                "task_id": task_response.status.message.taskId,
                "context_id": task_response.status.message.contextId,
                "timestamp": task_response.status.timestamp,
                "status": task_response.status.state,
                "message": {"Task completed successfully"},
            }
            assert result == expected_result
            mock_client.send_message.assert_called_once()
        else:
            msg = "task_response.status.message is None"
            raise ValueError(msg)


@pytest.mark.asyncio
async def test_handles_error_response() -> None:
    """Test that the a2a_tool properly handles receiving an error back from the server."""
    error_response = create_error_response()

    async with mock_a2a_tool(mock_agent_card(), error_response) as (tool, mock_client):
        result = await tool("Test query that will fail", None, None)

        # Verify the result is the expected error dictionary format
        expected_result = {
            "error": error_response.error.message,
            "code": error_response.error.code,
            "data": error_response.error.data,
        }
        assert result == expected_result
        mock_client.send_message.assert_called_once()
