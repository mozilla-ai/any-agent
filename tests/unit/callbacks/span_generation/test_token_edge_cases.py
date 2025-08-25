"""Tests for edge cases in token handling."""

from unittest.mock import MagicMock

import pytest

from any_agent.callbacks.span_generation.langchain import _LangchainSpanGeneration
from any_agent.callbacks.span_generation.google import _GoogleSpanGeneration
from any_agent.tracing.attributes import GenAI


def test_malformed_usage_data():
    """Test handling of malformed usage data."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock response with None usage
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Response"))]
    response.model_extra = {"usage": None}
    
    span_gen.after_llm_call(context, response)
    
    # Should use default values
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 0
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 0


def test_negative_token_values():
    """Test that negative token values are handled properly."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Response"))]
    
    # Create usage with specific attributes
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'completion_tokens_details'])
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    # Create completion details with negative reasoning tokens
    details = MagicMock(spec=['reasoning_tokens'])
    details.reasoning_tokens = -100  # Invalid negative
    usage.completion_tokens_details = details
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Negative values should not be included
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert GenAI.USAGE_REASONING_TOKENS not in call_args


def test_google_missing_usage_metadata():
    """Test Google span generation with missing usage metadata."""
    context = MagicMock()
    span_gen = _GoogleSpanGeneration()
    
    # Mock response without usage_metadata
    llm_response = MagicMock()
    llm_response.content.parts = [MagicMock(text="Response")]
    llm_response.usage_metadata = None
    
    span_gen.after_llm_call(context, llm_response=llm_response)
    
    # Should use default values
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 0
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 0


def test_very_large_token_values():
    """Test handling of very large token values."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Response"))]
    
    # Test with very large values
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'completion_tokens_details'])
    usage.prompt_tokens = 2**31 - 1  # Max int32
    usage.completion_tokens = 2**31 - 1
    # Create completion details with large reasoning tokens
    details = MagicMock(spec=['reasoning_tokens'])
    details.reasoning_tokens = 2**31 - 1
    usage.completion_tokens_details = details
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Should handle large values correctly
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 2**31 - 1
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 2**31 - 1
    assert call_args[GenAI.USAGE_REASONING_TOKENS] == 2**31 - 1