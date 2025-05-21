import datetime

from any_agent.tracing.otel_types import (
    Resource,
    SpanContext,
    SpanKind,
    Status,
)
from any_agent.tracing.trace import AgentSpan, AgentTrace


def test_agent_trace_duration_simple() -> None:
    # Create a span with the correct AGENT kind and name
    agent_span = AgentSpan(
        name="any_agent",
        kind=SpanKind.INTERNAL,
        parent=None,
        start_time=1000,
        end_time=2000,
        status=Status(),
        context=SpanContext(),
        attributes={"any_agent.run_id": "123"},
        links=[],
        events=[],
        resource=Resource(),
    )
    trace = AgentTrace(spans=[agent_span])
    expected = datetime.timedelta(seconds=(2000 - 1000) / 1_000_000_000)
    assert isinstance(trace.duration, datetime.timedelta)
    assert abs(trace.duration.total_seconds() - expected.total_seconds()) < 1e-9


def test_agent_trace_duration_from_sample(agent_trace: AgentTrace) -> None:
    """
    This test relies upon the sample trace that is saved in the sample_traces directory. If the content of that trace
    changes, this test will need to be updated
    (because it is using start and end times that were manually parsed from the trace)
    """
    # grab the start and end times from the last span in the trace
    start_time = agent_trace.spans[-1].start_time
    end_time = agent_trace.spans[-1].end_time
    assert start_time is not None
    assert end_time is not None
    expected_seconds = (end_time - start_time) / 1_000_000_000
    assert isinstance(agent_trace.duration, datetime.timedelta)
    assert abs(agent_trace.duration.total_seconds() - expected_seconds) < 1e-6
