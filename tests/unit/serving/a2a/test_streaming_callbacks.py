"""Test that streaming callbacks don't accumulate across requests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart

from any_agent import AgentConfig, AnyAgent
from any_agent.serving.a2a import A2AServingConfig
from any_agent.serving.a2a.agent_executor import AnyAgentExecutor
from any_agent.serving.a2a.context_manager import ContextManager
from any_agent.tools.final_output import FinalOutput
from any_agent.callbacks.base import Callback


class MockCallback(Callback):
    """Mock callback for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
    
    def before_tool_execution(self, context, *args, **kwargs):
        self.call_count += 1
        return context


@pytest.mark.asyncio
async def test_streaming_callbacks_no_accumulation():
    """Test that callbacks don't accumulate when stream_tool_usage is True."""
    
    # Create agent with initial callbacks
    initial_callback = MockCallback("initial")
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Test agent",
            output_type=FinalOutput,
            callbacks=[initial_callback],
        )
    )
    
    # Create executor with streaming enabled
    context_manager = ContextManager(A2AServingConfig(stream_tool_usage=True))
    executor = AnyAgentExecutor(
        agent=agent,
        context_manager=context_manager,
        stream_tool_usage=True,
    )
    
    # Mock the context and event queue
    mock_context = MagicMock(spec=RequestContext)
    mock_context.get_user_input.return_value = "test query"
    mock_context.current_task = None
    mock_context.message = MagicMock()
    mock_context.message.context_id = "test-context-123"
    
    mock_event_queue = AsyncMock(spec=EventQueue)
    
    # Track callback counts before each request
    callback_counts = []
    
    # Simulate multiple requests
    for i in range(3):
        # Record initial state
        initial_count = len(agent.config.callbacks)
        callback_counts.append(initial_count)
        
        # Mock agent.run_async to avoid actual execution
        with patch.object(agent, 'run_async', new_callable=AsyncMock) as mock_run:
            # Create a mock trace that includes FinalOutput
            mock_trace = MagicMock()
            mock_trace.final_output = FinalOutput(
                output="test response",
                task_status="complete"
            )
            mock_run.return_value = mock_trace
            
            # Execute the request
            await executor.execute(mock_context, mock_event_queue)
        
        # Verify callbacks were restored
        final_count = len(agent.config.callbacks)
        assert final_count == initial_count, (
            f"Request {i+1}: Callbacks not restored properly. "
            f"Expected {initial_count}, got {final_count}"
        )
    
    # Verify callback count remained constant across all requests
    assert all(count == callback_counts[0] for count in callback_counts), (
        f"Callback count varied across requests: {callback_counts}"
    )
    
    # The initial callback should still be there
    assert len(agent.config.callbacks) == 1
    assert agent.config.callbacks[0] == initial_callback



@pytest.mark.asyncio
async def test_streaming_callbacks_with_exception():
    """Test that callbacks are restored even when an exception occurs."""
    
    # Create agent
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Test agent",
            output_type=FinalOutput,
            callbacks=[],
        )
    )
    
    # Create executor with streaming enabled
    context_manager = ContextManager(A2AServingConfig(stream_tool_usage=True))
    executor = AnyAgentExecutor(
        agent=agent,
        context_manager=context_manager,
        stream_tool_usage=True,
    )
    
    # Mock the context and event queue
    mock_context = MagicMock(spec=RequestContext)
    mock_context.get_user_input.return_value = "test query"
    mock_context.current_task = None
    mock_context.message = MagicMock()
    mock_context.message.context_id = "test-context-456"
    
    mock_event_queue = AsyncMock(spec=EventQueue)
    
    initial_count = len(agent.config.callbacks)
    
    # Mock agent.run_async to raise an exception
    with patch.object(agent, 'run_async', new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = Exception("Test error")
        
        # Execute should handle the exception
        with pytest.raises(Exception, match="Test error"):
            await executor.execute(mock_context, mock_event_queue)
    
    # Verify callbacks were still restored
    final_count = len(agent.config.callbacks)
    assert final_count == initial_count, (
        f"Callbacks not restored after exception. "
        f"Expected {initial_count}, got {final_count}"
    )