from typing import Any, Dict, List
import json

from any_agent import AgentFramework
from loguru import logger
from any_agent.telemetry import TelemetryProcessor


class OpenAITelemetryProcessor(TelemetryProcessor):
    """Processor for OpenAI agent telemetry data."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.OPENAI

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

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> list:
        """Extract LLM calls and tool calls from OpenAI telemetry."""
        calls = []

        for span in telemetry:
            if "attributes" not in span:
                continue

            attributes = span.get("attributes", {})
            span_kind = attributes.get("openinference.span.kind", "")

            # Collect LLM interactions - look for direct message content first
            if span_kind == "LLM":
                # Initialize the LLM info dictionary
                span_info = {}

                # Try to get input message
                input_key = "llm.input_messages.1.message.content"  # User message is usually at index 1
                if input_key in attributes:
                    span_info["input"] = attributes[input_key]

                # Try to get output message directly
                output_content = None
                # Try in multiple possible locations
                for key in [
                    "llm.output_messages.0.message.content",
                    "llm.output_messages.0.message.contents.0.message_content.text",
                ]:
                    if key in attributes:
                        output_content = attributes[key]
                        break

                # If we found direct output content, use it
                if output_content:
                    span_info["output"] = output_content
                    calls.append(span_info)
            elif span_kind == "TOOL":
                tool_name = attributes.get("tool.name", "Unknown tool")
                tool_output = attributes.get("output.value", "")

                span_info = {
                    "tool_name": tool_name,
                    "input": attributes.get("input.value", ""),
                    "output": tool_output,
                    # Can't add status yet because it isn't being set by openinference
                    # "status": span.get("status", {}).get("status_code"),
                }
                span_info["input"] = json.loads(span_info["input"])

                calls.append(span_info)

        return calls
