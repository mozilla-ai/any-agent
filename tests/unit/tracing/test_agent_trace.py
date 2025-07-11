from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace
from any_agent.tracing.attributes import GenAI
from any_agent.tracing.otel_types import Resource, SpanContext, SpanKind, Status


def create_llm_span(input_tokens: int = 100, output_tokens: int = 50) -> AgentSpan:
    """Create a mock LLM span with token usage."""
    return AgentSpan(
        name=f"call_llm {DEFAULT_SMALL_MODEL_ID}",
        kind=SpanKind.INTERNAL,
        status=Status(),
        context=SpanContext(span_id=123),
        attributes={
            GenAI.OPERATION_NAME: "call_llm",
            GenAI.USAGE_INPUT_TOKENS: input_tokens,
            GenAI.USAGE_OUTPUT_TOKENS: output_tokens,
        },
        links=[],
        events=[],
        resource=Resource(),
    )


def test_tokens_and_cost_properties_are_cached() -> None:
    """Test that tokens and cost properties are cached after first access."""
    trace = AgentTrace()
    trace.add_span(create_llm_span(input_tokens=100, output_tokens=50))

    # First access - should compute and cache
    tokens1 = trace.tokens
    cost1 = trace.cost

    # Second access - should return cached objects
    tokens2 = trace.tokens
    cost2 = trace.cost

    assert tokens1 is tokens2  # Same object reference indicates caching
    assert cost1 is cost2  # Same object reference indicates caching
    assert "tokens" in trace.__dict__
    assert "cost" in trace.__dict__


def test_add_span_invalidates_cache() -> None:
    """Test that adding a span invalidates both tokens and cost caches."""
    trace = AgentTrace()
    trace.add_span(create_llm_span(input_tokens=100, output_tokens=50))

    # Cache the properties
    _ = trace.tokens
    _ = trace.cost
    assert "tokens" in trace.__dict__
    assert "cost" in trace.__dict__

    # Add another span - should invalidate cache
    trace.add_span(create_llm_span(input_tokens=200, output_tokens=75))

    assert "tokens" not in trace.__dict__
    assert "cost" not in trace.__dict__

    # Verify new calculations are correct
    tokens = trace.tokens
    assert tokens.input_tokens == 300
    assert tokens.output_tokens == 125


def test_invalidate_cache_method() -> None:
    """Test that _invalidate_tokens_and_cost_cache clears both caches."""
    trace = AgentTrace()
    trace.add_span(create_llm_span())

    # Cache both properties
    _ = trace.tokens
    _ = trace.cost
    assert "tokens" in trace.__dict__
    assert "cost" in trace.__dict__

    # Manually invalidate cache
    trace._invalidate_tokens_and_cost_cache()

    assert "tokens" not in trace.__dict__
    assert "cost" not in trace.__dict__


def test_spans_to_messages_handles_empty_spans() -> None:
    """Test that spans_to_messages handles traces with no spans."""
    empty_trace = AgentTrace()
    messages = empty_trace.spans_to_messages()

    assert isinstance(messages, list)
    assert len(messages) == 0
