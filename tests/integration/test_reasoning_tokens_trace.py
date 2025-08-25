"""Integration test for reasoning tokens in agent traces."""

from unittest.mock import MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tracing.attributes import GenAI


def test_reasoning_tokens_appear_in_trace():
    """Test that reasoning tokens from o1/Gemini models appear in traces."""
    
    # Mock response with reasoning tokens
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="This is a thoughtful response",
                tool_calls=None
            )
        )
    ]
    
    # Mock usage with reasoning tokens
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.completion_tokens_details = MagicMock(reasoning_tokens=300)
    mock_response.model_extra = {"usage": mock_usage}
    
    with patch("litellm.acompletion", return_value=mock_response):
        agent = AnyAgent.create(
            AgentFramework.LANGCHAIN,
            AgentConfig(
                model_id="openai/o1-preview",
                instructions="You are a thoughtful assistant."
            )
        )
        
        trace = agent.run("Explain quantum computing")
        
        # Find the LLM call span
        llm_spans = [
            span for span in trace.spans 
            if span.attributes.get(GenAI.OPERATION_NAME) == "call_llm"
        ]
        
        assert len(llm_spans) > 0
        llm_span = llm_spans[0]
        
        # Verify all token types are captured
        assert llm_span.attributes[GenAI.USAGE_INPUT_TOKENS] == 100
        assert llm_span.attributes[GenAI.USAGE_OUTPUT_TOKENS] == 50
        assert llm_span.attributes[GenAI.USAGE_REASONING_TOKENS] == 300


def test_google_thinking_tokens_in_trace():
    """Test that Google Gemini thinking tokens appear in traces."""
    
    # Mock Google response with thinking budget
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.parts = [MagicMock(text="Thoughtful Gemini response")]
    mock_response.content = mock_content
    
    # Mock usage metadata with thoughts_token_count
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 150
    mock_usage.candidates_token_count = 75
    mock_usage.thoughts_token_count = 400  # Gemini thinking tokens
    mock_response.usage_metadata = mock_usage
    
    # We need to mock the Google ADK Model's generate_content_async
    with patch("google.adk.models.lite_llm.LiteLlm.generate_content_async", return_value=mock_response):
        agent = AnyAgent.create(
            AgentFramework.GOOGLE,
            AgentConfig(
                model_id="gemini/gemini-2.0-flash",
                instructions="You are a thoughtful assistant."
            )
        )
        
        trace = agent.run("Solve this complex problem")
        
        # Find LLM call spans
        llm_spans = [
            span for span in trace.spans
            if span.attributes.get(GenAI.OPERATION_NAME) == "call_llm"
        ]
        
        assert len(llm_spans) > 0
        llm_span = llm_spans[0]
        
        # Verify thinking tokens are captured
        assert llm_span.attributes[GenAI.USAGE_INPUT_TOKENS] == 150
        assert llm_span.attributes[GenAI.USAGE_OUTPUT_TOKENS] == 75
        assert llm_span.attributes[GenAI.USAGE_REASONING_TOKENS] == 400


def test_no_reasoning_tokens_for_regular_models():
    """Test that regular models don't have reasoning tokens in traces."""
    
    # Mock response without reasoning tokens
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Regular response",
                tool_calls=None
            )
        )
    ]
    
    # Mock usage without reasoning tokens
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 80
    mock_usage.completion_tokens = 40
    # No completion_tokens_details
    mock_response.model_extra = {"usage": mock_usage}
    
    with patch("litellm.acompletion", return_value=mock_response):
        agent = AnyAgent.create(
            AgentFramework.LANGCHAIN,
            AgentConfig(
                model_id="gpt-4",
                instructions="You are a helpful assistant."
            )
        )
        
        trace = agent.run("Hello")
        
        # Find LLM call span
        llm_spans = [
            span for span in trace.spans
            if span.attributes.get(GenAI.OPERATION_NAME) == "call_llm"
        ]
        
        assert len(llm_spans) > 0
        llm_span = llm_spans[0]
        
        # Verify no reasoning tokens attribute
        assert llm_span.attributes[GenAI.USAGE_INPUT_TOKENS] == 80
        assert llm_span.attributes[GenAI.USAGE_OUTPUT_TOKENS] == 40
        assert GenAI.USAGE_REASONING_TOKENS not in llm_span.attributes