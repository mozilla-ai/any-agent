import contextlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from any_agent import AgentFramework
from any_agent.logging import logger
from any_agent.telemetry import TelemetryProcessor


class GoogleTelemetryProcessor(TelemetryProcessor):
    """Processor for Google ADK telemetry data."""

    def extract_interaction(
        self,
        span: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any]]:
        """Extract interaction details from a span."""
        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        if span_kind == "LLM":
            return "LLM", self._extract_llm_interaction(span)
        if span_kind == "TOOL":
            return "TOOL", self._extract_tool_interaction(span)
        if span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        return "UNKNOWN", {}

    def _extract_hypothesis_answer(self, trace: Sequence[Mapping[str, Any]]) -> str:
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
                    return str(span["attributes"][output_key])
        logger.warning("No agent final answer found in trace")
        return "NO FINAL ANSWER FOUND"

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.GOOGLE

    def _extract_llm_interaction(self, span: Mapping[str, Any]) -> dict[str, Any]:
        attributes = span.get("attributes", {})
        span_info = {}

        input_key = "input.value"
        if input_key in attributes:
            span_info["input"] = attributes[input_key]

        output_key = "llm.output_messages"
        if output_key in attributes:
            span_info["output"] = json.loads(attributes[output_key])

        return span_info

    def _extract_tool_interaction(self, span: Mapping[str, Any]) -> dict[str, Any]:
        attributes = span.get("attributes", {})
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

    def _extract_agent_interaction(self, span: Mapping[str, Any]) -> dict[str, Any]:
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

    def _extract_chain_interaction(self, span: Mapping[str, Any]) -> dict[str, Any]:
        """Extract information from a CHAIN span."""
        return {}
