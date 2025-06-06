from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .exporter import _ConsoleExporter

TRACE_PROVIDER = TracerProvider()
trace.set_tracer_provider(TRACE_PROVIDER)
TRACE_PROVIDER.add_span_processor(SimpleSpanProcessor(_ConsoleExporter()))


def drop_console_exporter() -> None:
    """Disable printing traces to the console."""
    with TRACE_PROVIDER._active_span_processor._lock:
        TRACE_PROVIDER._active_span_processor._span_processors = tuple(
            p
            for p in TRACE_PROVIDER._active_span_processor._span_processors
            if not isinstance(getattr(p, "span_exporter", None), _ConsoleExporter)
        )
