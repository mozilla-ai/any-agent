# mypy: disable-error-code="no-untyped-def"
from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context


def _span_end(context: Context) -> Context:
    context.current_span.end()
    context.trace.add_span(context.current_span)
    return context


class SpanEndCallback(Callback):
    """End the current span and add it to the corresponding `AgentTrace`."""

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        return _span_end(context)

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        return _span_end(context)
