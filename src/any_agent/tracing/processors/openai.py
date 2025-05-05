import contextlib
import json
from typing import TYPE_CHECKING, Any

from any_agent import AgentFramework
from any_agent.logging import logger
from any_agent.tracing.processors.base import TracingProcessor

if TYPE_CHECKING:
    from any_agent.tracing.trace import AgentSpan, AgentTrace


class OpenAITracingProcessor(TracingProcessor):
    """Processor for OpenAI agent trace."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.OPENAI

    def _extract_hypothesis_answer(self, trace: "AgentTrace") -> str:
        for span in reversed(trace.spans):
            # Looking for the final response that has the summary answer
            if span.attributes.get("openinference.span.kind") == "LLM":
                output_key = (
                    "llm.output_messages.0.message.contents.0.message_content.text"
                )
                if output_key in span.attributes:
                    return str(span.attributes[output_key])
        logger.warning("No agent final answer found in trace")
        return "NO FINAL ANSWER FOUND"

    def _extract_llm_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        span_info = {}

        input_key = "llm.input_messages.1.message.content"
        if input_key in attributes:
            span_info["input"] = attributes[input_key]

        output_content = None
        for key in [
            "llm.output_messages.0.message.content",
            "llm.output_messages.0.message.contents.0.message_content.text",
        ]:
            if key in attributes:
                output_content = attributes[key]
                break

        if output_content:
            span_info["output"] = output_content

        return span_info

    def _extract_tool_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        tool_name = attributes.get("tool.name", "Unknown tool")
        tool_output = attributes.get("output.value", "")

        span_info = {
            "tool_name": tool_name,
            "input": attributes.get("input.value", ""),
            "output": tool_output,
        }

        with contextlib.suppress(json.JSONDecodeError):
            span_info["input"] = json.loads(span_info["input"])

        return span_info

    def _extract_agent_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract information from an AGENT span."""
        span_info: dict[str, Any] = {
            "type": "agent",
            "workflow": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
        }

        # Add any additional attributes that might be useful
        if "service.name" in span.resource.attributes:
            span_info["service"] = span.resource.attributes["service.name"]

        return span_info

    def _extract_chain_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract information from a CHAIN span."""
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)

        span_info: dict[str, Any] = {
            "type": "chain",
            "workflow": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
        }

        # Extract input and output values
        input_value = attributes.get("input.value", "")
        output_value = attributes.get("output.value", "")

        # Try to parse JSON if available
        try:
            span_info["input"] = json.loads(input_value)
        except (json.JSONDecodeError, TypeError):
            span_info["input"] = input_value

        try:
            span_info["output"] = json.loads(output_value)
        except (json.JSONDecodeError, TypeError):
            span_info["output"] = output_value

        # Add service name if available
        if "service.name" in span.resource.attributes:
            span_info["service"] = span.resource.attributes["service.name"]

        return span_info
