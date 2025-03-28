from typing import Any, Dict, List
import json

from any_agent import AgentFramework

from any_agent.telemetry import TelemetryProcessor


class SmolagentsTelemetryProcessor(TelemetryProcessor):
    """Processor for SmoL Agents telemetry data."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def extract_hypothesis_answer(self, trace: List[Dict[str, Any]]) -> str:
        for span in reversed(trace):
            if span["attributes"]["openinference.span.kind"] == "AGENT":
                content = span["attributes"]["output.value"]
                return content

        raise ValueError("No agent final answer found in trace")

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> List[Dict]:
        """Extract LLM calls and tool calls from SmoL Agents telemetry."""
        calls = []

        for span in telemetry:
            # Skip spans without attributes
            if "attributes" not in span:
                continue

            attributes = span["attributes"]

            # Extract tool information
            if "tool.name" in attributes or span.get("name", "").startswith(
                "SimpleTool"
            ):
                tool_info = {
                    "tool_name": attributes.get(
                        "tool.name", span.get("name", "Unknown tool")
                    ),
                    "status": "success"
                    if span.get("status", {}).get("status_code") == "OK"
                    else "error",
                    "error": span.get("status", {}).get("description", None),
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

                calls.append(tool_info)

            # Extract LLM calls to see reasoning
            elif "LiteLLMModel.__call__" in span.get("name", ""):
                # The LLM output may be in different places depending on the implementation
                output_content = None

                # Try to get the output from the llm.output_messages.0.message.content attribute
                if "llm.output_messages.0.message.content" in attributes:
                    output_content = attributes["llm.output_messages.0.message.content"]

                # Or try to parse it from the output.value as JSON
                elif "output.value" in attributes:
                    try:
                        output_value = json.loads(attributes["output.value"])
                        if "content" in output_value:
                            output_content = output_value["content"]
                    except (json.JSONDecodeError, TypeError):
                        pass

                if output_content:
                    calls.append(
                        {
                            "model": attributes.get("llm.model_name", "Unknown model"),
                            "output": output_content,
                            "type": "reasoning",
                        }
                    )

        return calls
