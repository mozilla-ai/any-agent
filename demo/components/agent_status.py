from collections.abc import Callable
from typing import Any

from any_agent.callbacks import Callback, Context
from any_agent.tracing.attributes import GenAI


class StreamlitStatusCallback(Callback):
    """Callback to update Streamlit status with agent progress."""

    def __init__(self, status_callback: Callable[[str], None]):
        self.status_callback = status_callback

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        """Update status after LLM calls."""
        span = context.current_span
        input_value = span.attributes.get(GenAI.INPUT_MESSAGES, "")
        output_value = span.attributes.get(GenAI.OUTPUT, "")

        self._update_status(span.name, input_value, output_value)
        return context

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        """Update status after tool executions."""
        span = context.current_span
        input_value = span.attributes.get(GenAI.TOOL_ARGS, "")
        output_value = span.attributes.get(GenAI.OUTPUT, "")

        self._update_status(span.name, input_value, output_value)
        return context

    def _update_status(self, step_name: str, input_value: str, output_value: str):
        """Update the Streamlit status with formatted information."""
        if input_value:
            try:
                import json

                parsed_input = json.loads(input_value)
                if isinstance(parsed_input, list) and len(parsed_input) > 0:
                    input_value = str(parsed_input[-1])
            except Exception:
                pass

        if output_value:
            try:
                import json

                parsed_output = json.loads(output_value)
                if isinstance(parsed_output, list) and len(parsed_output) > 0:
                    output_value = str(parsed_output[-1])
            except Exception:
                pass

        max_length = 800
        if len(input_value) > max_length:
            input_value = f"[Truncated]...{input_value[-max_length:]}"
        if len(output_value) > max_length:
            output_value = f"[Truncated]...{output_value[-max_length:]}"

        if input_value or output_value:
            message = f"Step: {step_name}\n"
            if input_value:
                message += f"Input: {input_value}\n"
            if output_value:
                message += f"Output: {output_value}"
        else:
            message = f"Step: {step_name}"

        self.status_callback(message)


def export_logs(agent: Any, callback: Callable[[str], None]) -> None:
    """Add a Streamlit status callback to the agent.

    This function adds a custom callback to the agent that will update
    the Streamlit status with progress information during agent execution.
    """
    status_callback = StreamlitStatusCallback(callback)

    if agent.config.callbacks is None:
        agent.config.callbacks = []
    agent.config.callbacks.append(status_callback)
