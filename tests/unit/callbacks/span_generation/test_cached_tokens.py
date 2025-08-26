"""Tests for cached tokens in span generation."""

from unittest.mock import MagicMock

import pytest

from any_agent.callbacks.span_generation.langchain import _LangchainSpanGeneration
from any_agent.tracing.attributes import GenAI


def test_langchain_with_cached_tokens():
    """Test LangChain span generation with OpenAI cached tokens."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock response with cached tokens
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Cached response"))]
    
    # Mock usage with cached tokens
    usage = MagicMock()
    usage.prompt_tokens = 200
    usage.completion_tokens = 50
    usage.prompt_tokens_details = MagicMock(cached_tokens=150)
    # Ensure completion_tokens_details doesn't exist to avoid reasoning_tokens extraction
    if hasattr(usage, 'completion_tokens_details'):
        del usage.completion_tokens_details
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Verify cached tokens were captured
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 200
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 50
    assert call_args[GenAI.USAGE_CACHED_TOKENS] == 150


def test_langchain_without_cached_tokens():
    """Test that cached tokens are not set when zero."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock response without cached tokens
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Regular response"))]
    
    # Create usage object without prompt_tokens_details
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens'])
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Verify cached tokens not included
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert GenAI.USAGE_CACHED_TOKENS not in call_args


def test_edge_case_missing_prompt_details():
    """Test handling when prompt_tokens_details exists but has no cached_tokens."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Response"))]
    
    # Create usage with prompt_tokens_details but no cached_tokens
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'prompt_tokens_details'])
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = MagicMock(spec=[])  # Empty spec, no attributes
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Should not crash, cached tokens should not be set
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert GenAI.USAGE_CACHED_TOKENS not in call_args