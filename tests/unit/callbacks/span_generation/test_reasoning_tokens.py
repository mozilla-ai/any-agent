"""Tests for reasoning/thinking tokens in span generation."""

from unittest.mock import MagicMock, patch

import pytest

from any_agent.callbacks.span_generation.base import _SpanGeneration
from any_agent.callbacks.span_generation.google import _GoogleSpanGeneration
from any_agent.callbacks.span_generation.langchain import _LangchainSpanGeneration
from any_agent.tracing.attributes import GenAI


def test_base_span_generation_with_reasoning_tokens():
    """Test that base span generation handles reasoning tokens correctly."""
    context = MagicMock()
    span_gen = _SpanGeneration()
    
    # Test with reasoning tokens
    span_gen._set_llm_output(
        context, 
        output="Response with reasoning",
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=200
    )
    
    context.current_span.set_attributes.assert_called_with({
        GenAI.OUTPUT: "Response with reasoning",
        GenAI.OUTPUT_TYPE: "text",
        GenAI.USAGE_INPUT_TOKENS: 100,
        GenAI.USAGE_OUTPUT_TOKENS: 50,
        GenAI.USAGE_REASONING_TOKENS: 200,
    })


def test_base_span_generation_without_reasoning_tokens():
    """Test that reasoning tokens are not set when zero."""
    context = MagicMock()
    span_gen = _SpanGeneration()
    
    # Test without reasoning tokens (default)
    span_gen._set_llm_output(
        context,
        output="Response without reasoning", 
        input_tokens=100,
        output_tokens=50
    )
    
    context.current_span.set_attributes.assert_called_with({
        GenAI.OUTPUT: "Response without reasoning",
        GenAI.OUTPUT_TYPE: "text",
        GenAI.USAGE_INPUT_TOKENS: 100,
        GenAI.USAGE_OUTPUT_TOKENS: 50,
    })


def test_google_span_generation_with_thinking_tokens():
    """Test Google span generation extracts thoughts_token_count."""
    context = MagicMock()
    span_gen = _GoogleSpanGeneration()
    
    # Mock Google LLM response with thinking tokens
    llm_response = MagicMock()
    llm_response.content.parts = [MagicMock(text="Gemini response")]
    
    # Mock usage metadata with thoughts_token_count
    usage_metadata = MagicMock()
    usage_metadata.prompt_token_count = 150
    usage_metadata.candidates_token_count = 75
    usage_metadata.thoughts_token_count = 300  # Gemini thinking tokens
    llm_response.usage_metadata = usage_metadata
    
    span_gen.after_llm_call(context, llm_response=llm_response)
    
    # Verify _set_llm_output was called with thinking tokens
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 150
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 75
    assert call_args[GenAI.USAGE_REASONING_TOKENS] == 300


def test_google_span_generation_without_thinking_tokens():
    """Test Google span generation when no thinking tokens."""
    context = MagicMock()
    span_gen = _GoogleSpanGeneration()
    
    # Mock response without thoughts_token_count
    llm_response = MagicMock()
    llm_response.content.parts = [MagicMock(text="Regular response")]
    
    usage_metadata = MagicMock()
    usage_metadata.prompt_token_count = 100
    usage_metadata.candidates_token_count = 50
    # No thoughts_token_count attribute
    del usage_metadata.thoughts_token_count
    llm_response.usage_metadata = usage_metadata
    
    span_gen.after_llm_call(context, llm_response=llm_response)
    
    # Verify reasoning tokens not included when zero
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert GenAI.USAGE_REASONING_TOKENS not in call_args


def test_langchain_span_generation_with_openai_reasoning():
    """Test LangChain span generation with OpenAI reasoning tokens."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock litellm ModelResponse with reasoning tokens
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="o1 response"))]
    
    # Mock usage with completion_tokens_details
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'completion_tokens_details'])
    usage.prompt_tokens = 200
    usage.completion_tokens = 100
    details = MagicMock(spec=['reasoning_tokens'])
    details.reasoning_tokens = 500
    usage.completion_tokens_details = details
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Verify reasoning tokens were captured
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 200
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 100  
    assert call_args[GenAI.USAGE_REASONING_TOKENS] == 500


def test_langchain_with_all_token_types():
    """Test LangChain with reasoning and cached tokens."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock response with all token types
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Full response"))]
    
    # Mock comprehensive usage
    usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'completion_tokens_details', 'prompt_tokens_details'])
    usage.prompt_tokens = 150
    usage.completion_tokens = 80
    
    completion_details = MagicMock(spec=['reasoning_tokens'])
    completion_details.reasoning_tokens = 200
    usage.completion_tokens_details = completion_details
    
    prompt_details = MagicMock(spec=['cached_tokens'])
    prompt_details.cached_tokens = 100
    usage.prompt_tokens_details = prompt_details
    
    response.model_extra = {"usage": usage}
    
    span_gen.after_llm_call(context, response)
    
    # Verify all token types captured
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_TOKENS] == 150
    assert call_args[GenAI.USAGE_OUTPUT_TOKENS] == 80
    assert call_args[GenAI.USAGE_REASONING_TOKENS] == 200
    assert call_args[GenAI.USAGE_CACHED_TOKENS] == 100


def test_langchain_llmresult_with_reasoning_tokens():
    """Test LangChain LLMResult format with reasoning tokens."""
    context = MagicMock()
    span_gen = _LangchainSpanGeneration()
    
    # Mock LangChain LLMResult
    response = MagicMock(spec=['generations', 'llm_output'])
    response.generations = [[MagicMock(text="Generated text")]]
    
    # Mock token usage in llm_output
    token_usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'completion_tokens_details'])
    token_usage.prompt_tokens = 150
    token_usage.completion_tokens = 80
    details = MagicMock(spec=['reasoning_tokens'])
    details.reasoning_tokens = 250
    token_usage.completion_tokens_details = details
    response.llm_output = {"token_usage": token_usage}
    
    span_gen.after_llm_call(context, response)
    
    # Verify all token types captured
    context.current_span.set_attributes.assert_called()
    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_REASONING_TOKENS] == 250