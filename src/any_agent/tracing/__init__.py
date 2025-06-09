# mypy: disable-error-code="attr-defined"
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .exporter import _ConsoleExporter

TRACE_PROVIDER = trace.get_tracer_provider()
if isinstance(TRACE_PROVIDER, trace.ProxyTracerProvider):
    TRACE_PROVIDER = TracerProvider()
    trace.set_tracer_provider(TRACE_PROVIDER)


def enable_console_traces() -> None:
    """Enable printing traces to the console."""
    has_console_exporter = any(
        isinstance(getattr(p, "span_exporter", None), _ConsoleExporter)
        for p in TRACE_PROVIDER._active_span_processor._span_processors
    )
    if not has_console_exporter:
        TRACE_PROVIDER.add_span_processor(SimpleSpanProcessor(_ConsoleExporter()))


def disable_console_traces() -> None:
    """Disable printing traces to the console."""
    with TRACE_PROVIDER._active_span_processor._lock:
        TRACE_PROVIDER._active_span_processor._span_processors = tuple(
            p
            for p in TRACE_PROVIDER._active_span_processor._span_processors
            if not isinstance(getattr(p, "span_exporter", None), _ConsoleExporter)
        )


enable_console_traces()
