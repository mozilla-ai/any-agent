# mypy: disable-error-code="no-untyped-def"
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

from any_agent.callbacks.base import Callback

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from any_agent.tracing.agent_trace import AgentTrace


def _span_end(context: dict[str, Any], operation_name: str) -> None:
    trace_id = get_current_span().get_span_context().trace_id
    current_span: Span = context[f"{operation_name}-{trace_id}"]
    current_span.end()

    trace: AgentTrace = context["running_traces"][trace_id]
    trace.add_span(current_span)


class SpanEndCallback(Callback):
    """End the current span and add it to the corresponding `AgentTrace`."""

    def after_llm_call(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        _span_end(context, "call_llm")
        return context

    def after_tool_execution(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        _span_end(context, "execute_tool")
        return context
