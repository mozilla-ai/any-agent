from typing import Any, Dict, List
import json
from any_agent import AgentFramework
from langchain_core.messages import BaseMessage


from any_agent.telemetry import TelemetryProcessor


class LangchainTelemetryProcessor(TelemetryProcessor):
    """Processor for Langchain agent telemetry data."""

    def _get_agent_framework(self) -> AgentFramework:
        return AgentFramework.LANGCHAIN

    def extract_hypothesis_answer(self, trace: List[Dict[str, Any]]) -> str:
        for span in reversed(trace):
            if span["attributes"]["openinference.span.kind"] == "AGENT":
                content = span["attributes"]["output.value"]
                # Extract content from serialized langchain message
                message = json.loads(content)["messages"][0]
                message = self.parse_generic_key_value_string(message)
                base_message = BaseMessage(content=message["content"], type="AGENT")
                # Use the interpreted string for printing
                final_text = base_message.text()
                # Either decode escape sequences if they're present
                try:
                    final_text = final_text.encode().decode("unicode_escape")
                except UnicodeDecodeError:
                    # If that fails, the escape sequences might already be interpreted
                    pass
                return final_text

        raise ValueError("No agent final answer found in trace")

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> List[Dict]:
        """Extract LLM calls and tool calls from LangChain telemetry."""
        calls = []

        for span in telemetry:
            if "attributes" not in span:
                continue

            attributes = span.get("attributes", {})
            span_kind = attributes.get("openinference.span.kind", "")

            # Collect LLM calls
            if (
                span_kind == "LLM"
                and "llm.output_messages.0.message.content" in attributes
            ):
                llm_info = {
                    "model": attributes.get("llm.model_name", "Unknown model"),
                    "input": attributes.get("llm.input_messages.0.message.content", ""),
                    "output": attributes.get(
                        "llm.output_messages.0.message.content", ""
                    ),
                    "type": "reasoning",
                }
                calls.append(llm_info)

            # Try to find tool calls
            if "tool.name" in attributes or span.get("name", "").endswith("Tool"):
                tool_info = {
                    "tool_name": attributes.get(
                        "tool.name", span.get("name", "Unknown tool")
                    ),
                    "status": "success"
                    if span.get("status", {}).get("status_code") == "OK"
                    else "error",
                    "error": span.get("status", {}).get("description", None),
                }

                if "input.value" in attributes:
                    try:
                        input_value = json.loads(attributes["input.value"])
                        tool_info["input"] = input_value
                    except Exception:
                        tool_info["input"] = attributes["input.value"]

                if "output.value" in attributes:
                    tool_info["output"] = self.parse_generic_key_value_string(
                        json.loads(attributes["output.value"])["output"]
                    )["content"]

                calls.append(tool_info)

        return calls
