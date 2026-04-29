"""Re-export OTel value types from `tinyagent`."""

from tinyagent.tracing.otel_types import (
    AttributeValue,
    Event,
    Link,
    Resource,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    TraceState,
)

__all__ = [
    "AttributeValue",
    "Event",
    "Link",
    "Resource",
    "SpanContext",
    "SpanKind",
    "Status",
    "StatusCode",
    "TraceFlags",
    "TraceState",
]
