# mypy: disable-error-code="attr-defined"
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

TRACE_PROVIDER = trace.get_tracer_provider()
if isinstance(TRACE_PROVIDER, trace.ProxyTracerProvider):
    TRACE_PROVIDER = TracerProvider()
    trace.set_tracer_provider(TRACE_PROVIDER)
