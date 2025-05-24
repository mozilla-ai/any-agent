from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

TRACE_PROVIDER = TracerProvider()
trace.set_tracer_provider(TRACE_PROVIDER)
