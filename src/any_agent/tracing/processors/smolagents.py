import json
from typing import TYPE_CHECKING, Any

from any_agent import AgentFramework
from any_agent.tracing.otel_types import StatusCode
from any_agent.tracing.processors.base import TracingProcessor

if TYPE_CHECKING:
    from any_agent.tracing.trace import AgentSpan, AgentTrace


class SmolagentsTracingProcessor(TracingProcessor):
    """Processor for SmoL Agents trace."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def _extract_hypothesis_answer(self, trace: "AgentTrace") -> str:
        for span in reversed(trace.spans):
            if span.attributes["openinference.span.kind"] == "AGENT":
                return str(span.attributes["output.value"])

        msg = "No agent final answer found in trace"
        raise ValueError(msg)

    def _extract_llm_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract LLM interaction details from a span."""
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        span_info = {
            "model": attributes.get("llm.model_name", "Unknown model"),
            "type": "reasoning",
        }

        # Try to get the input from various possible attribute locations
        if "llm.input_messages.0.message.content" in attributes:
            span_info["input"] = attributes["llm.input_messages.0.message.content"]
        elif "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                span_info["input"] = input_value.get("content", input_value)
            except (json.JSONDecodeError, TypeError):
                span_info["input"] = attributes["input.value"]

        # Try to get the output from various possible attribute locations
        output_content = None
        if "llm.output_messages.0.message.content" in attributes:
            output_content = attributes["llm.output_messages.0.message.content"]
        elif "output.value" in attributes:
            try:
                output_value = json.loads(attributes["output.value"])
                output_content = output_value.get("content", output_value)
            except (json.JSONDecodeError, TypeError):
                output_content = attributes["output.value"]

        if output_content:
            span_info["output"] = output_content

        return span_info

    def _extract_tool_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract tool interaction details from a span."""
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        tool_info = {
            "tool_name": attributes.get("tool.name", span.name),
            "status": "success"
            if span.status.status_code is StatusCode.OK
            else "error",
            "error": span.status.description,
        }

        # Extract input if available
        if "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                if "kwargs" in input_value:
                    # For SmoLAgents, the actual input is often in the kwargs field
                    tool_info["input"] = input_value["kwargs"]
                else:
                    tool_info["input"] = input_value
            except (json.JSONDecodeError, TypeError):
                tool_info["input"] = attributes["input.value"]

        # Extract output if available
        if "output.value" in attributes:
            try:
                # Try to parse JSON output
                output_value = (
                    json.loads(attributes["output.value"])
                    if isinstance(attributes["output.value"], str)
                    else attributes["output.value"]
                )
                tool_info["output"] = output_value
            except (json.JSONDecodeError, TypeError):
                tool_info["output"] = attributes["output.value"]
        else:
            tool_info["output"] = "No output found"

        return tool_info

    def _extract_chain_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract chain interaction details from a CHAIN span."""
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        status = span.status
        events = span.events

        chain_info: dict[str, Any] = {
            "type": "chain",
            "name": span.name,
            "status": "success" if status.status_code is StatusCode.OK else "error",
        }

        # Extract input if available
        if "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                chain_info["input"] = input_value
            except (json.JSONDecodeError, TypeError):
                chain_info["input"] = attributes["input.value"]

        # Extract error information if present
        if status.status_code == StatusCode.ERROR:
            chain_info["error"] = status.description

            # Try to extract more detailed error info from events
            for event in events:
                if event.name == "exception":
                    event_attrs = event.attributes
                    if event_attrs is None:
                        continue
                    if "exception.type" in event_attrs:
                        chain_info["error_type"] = event_attrs["exception.type"]
                    if "exception.message" in event_attrs:
                        chain_info["error_message"] = event_attrs["exception.message"]
                    break

        return chain_info

    def _extract_agent_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract agent interaction details from an AGENT span."""
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        status = span.status

        agent_info: dict[str, Any] = {
            "type": "agent",
            "name": span.name,
            "status": "success" if status.status_code == StatusCode.OK else "error",
        }

        # Extract input if available
        if "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                agent_info["input"] = input_value
            except (json.JSONDecodeError, TypeError):
                agent_info["input"] = attributes["input.value"]

        # Extract output (final answer) if available
        if "output.value" in attributes:
            agent_info["output"] = attributes["output.value"]

        # Extract additional metadata if available
        if "smolagents.max_steps" in attributes:
            agent_info["max_steps"] = attributes["smolagents.max_steps"]

        if "smolagents.tools_names" in attributes:
            agent_info["tools"] = attributes["smolagents.tools_names"]

        # Extract token usage if available
        token_counts = {}
        for key in [
            "llm.token_count.prompt",
            "llm.token_count.completion",
            "llm.token_count.total",
        ]:
            if key in attributes:
                token_name = key.split(".")[-1]
                token_counts[token_name] = attributes[key]

        if token_counts:
            agent_info["token_usage"] = token_counts

        return agent_info
