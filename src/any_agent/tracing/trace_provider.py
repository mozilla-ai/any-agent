from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

TRACE_PROVIDER = TracerProvider()
trace.set_tracer_provider(TRACE_PROVIDER)

# Global console exporter to prevent duplicates
# These will be set when the first agent with console=True is created
_GLOBAL_CONSOLE_EXPORTER = None
_GLOBAL_CONSOLE_PROCESSOR = None
