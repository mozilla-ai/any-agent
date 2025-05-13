import json
from typing import TYPE_CHECKING, Any

from openinference.semconv.trace import (
    SpanAttributes,
)

from any_agent import AgentFramework
from any_agent.logging import logger
from any_agent.tracing.otel_types import StatusCode
from any_agent.tracing.processors.base import TracingProcessor

if TYPE_CHECKING:
    from any_agent.tracing.trace import AgentSpan


class AgnoTracingProcessor(TracingProcessor):
    """Processor for Agno Agents trace."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.AGNO

    def _extract_llm_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract LLM interaction details from a span."""
        logger.debug(f"Received llm span [AGNO]: \n{span}")
        attributes = span.attributes
        if not attributes:
            msg = "Span attributes are empty"
            raise ValueError(msg)
        span_info = {
            "model": attributes.get(SpanAttributes.LLM_MODEL_NAME, "Unknown model"),
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
        logger.debug(f"Received tool span [AGNO]: \n{span}")
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
        # Apparently it's not used in Agno
        logger.debug(f"Received chain span [AGNO]: \n{span}")
        return {}

    def _extract_agent_interaction(self, span: "AgentSpan") -> dict[str, Any]:
        """Extract agent interaction details from an AGENT span."""
        logger.debug(f"Received agent span [AGNO]: \n{span}")
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

        if "agno.agent" in attributes:
            agent_info["agent_name"] = attributes["agno.agent"]
        if "agno.tools" in attributes:
            agent_info["tools"] = attributes["agno.agent.tools"]

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
