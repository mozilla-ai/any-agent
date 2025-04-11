from typing import Any, Dict, List
import json

from any_agent import AgentFramework
from any_agent.logging import logger
from any_agent.telemetry import TelemetryProcessor


class LlamaIndexTelemetryProcessor(TelemetryProcessor):
    """Processor for LlamaIndex agent telemetry data."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.LLAMAINDEX

    def extract_hypothesis_answer(self, trace: List[Dict[str, Any]]) -> str:
        for span in reversed(trace):
            # Looking for the final response that has the summary answer
            if (
                "attributes" in span
                and span.get("attributes", {}).get("openinference.span.kind") == "LLM"
            ):
                output_key = (
                    "llm.output_messages.0.message.contents.0.message_content.text"
                )
                if output_key in span["attributes"]:
                    return span["attributes"][output_key]
        logger.warning("No agent final answer found in trace")
        return "NO FINAL ANSWER FOUND"

    def _extract_llm_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        attributes = span.get("attributes", {})
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

    def _extract_tool_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        attributes = span.get("attributes", {})
        tool_name = attributes.get("tool.name", "Unknown tool")
        tool_output = attributes.get("output.value", "")

        span_info = {
            "tool_name": tool_name,
            "input": attributes.get("input.value", ""),
            "output": tool_output,
        }

        try:
            span_info["input"] = json.loads(span_info["input"])
        except json.JSONDecodeError:
            pass

        return span_info

    def _extract_agent_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information from an AGENT span."""

        span_info = {
            "type": "agent",
            "workflow": span.get("name", "Agent workflow"),
            "start_time": span.get("start_time"),
            "end_time": span.get("end_time"),
        }

        # Add any additional attributes that might be useful
        if "service.name" in span.get("resource", {}).get("attributes", {}):
            span_info["service"] = span["resource"]["attributes"]["service.name"]

        return span_info

    def _extract_chain_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information from a CHAIN span."""
        attributes = span.get("attributes", {})

        span_info = {
            "type": "chain",
            "workflow": span.get("name", "Chain workflow"),
            "start_time": span.get("start_time"),
            "end_time": span.get("end_time"),
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
        if "service.name" in span.get("resource", {}).get("attributes", {}):
            span_info["service"] = span["resource"]["attributes"]["service.name"]

        return span_info

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> list:
        """Extract LLM calls and tool calls from LlamaIndex telemetry."""
        calls = []

        for span in telemetry:
            calls.append(self.extract_interaction(span)[1])

        return calls

    def extract_interaction(self, span: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract interaction details from a span."""
        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        if span_kind == "LLM":
            return "LLM", self._extract_llm_interaction(span)
        elif span_kind == "TOOL":
            return "TOOL", self._extract_tool_interaction(span)
        elif span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        elif span_kind == "CHAIN":
            return "CHAIN", self._extract_chain_interaction(span)
        else:
            raise ValueError(
                f"Unknown span kind: {span_kind}. Expected 'LLM', 'TOOL', 'AGENT', or 'CHAIN'."
            )
