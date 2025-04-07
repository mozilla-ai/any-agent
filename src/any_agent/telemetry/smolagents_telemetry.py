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

    def _extract_llm_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract LLM interaction details from a span."""
        attributes = span.get("attributes", {})
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
                if "content" in input_value:
                    span_info["input"] = input_value["content"]
                else:
                    span_info["input"] = input_value
            except (json.JSONDecodeError, TypeError):
                span_info["input"] = attributes["input.value"]

        # Try to get the output from various possible attribute locations
        output_content = None
        if "llm.output_messages.0.message.content" in attributes:
            output_content = attributes["llm.output_messages.0.message.content"]
        elif "output.value" in attributes:
            try:
                output_value = json.loads(attributes["output.value"])
                if "content" in output_value:
                    output_content = output_value["content"]
                else:
                    output_content = output_value
            except (json.JSONDecodeError, TypeError):
                output_content = attributes["output.value"]

        if output_content:
            span_info["output"] = output_content

        return span_info

    def _extract_tool_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool interaction details from a span."""
        attributes = span.get("attributes", {})
        tool_info = {
            "tool_name": attributes.get("tool.name", span.get("name", "Unknown tool")),
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

        return tool_info

    def _extract_chain_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chain interaction details from a CHAIN span."""
        attributes = span.get("attributes", {})
        status = span.get("status", {})
        events = span.get("events", [])

        chain_info = {
            "type": "chain",
            "name": span.get("name", "Unknown chain"),
            "status": "success" if status.get("status_code") == "OK" else "error",
        }

        # Extract input if available
        if "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                chain_info["input"] = input_value
            except (json.JSONDecodeError, TypeError):
                chain_info["input"] = attributes["input.value"]

        # Extract error information if present
        if status.get("status_code") == "ERROR":
            chain_info["error"] = status.get("description", "Unknown error")

            # Try to extract more detailed error info from events
            for event in events:
                if event.get("name") == "exception":
                    event_attrs = event.get("attributes", {})
                    if "exception.type" in event_attrs:
                        chain_info["error_type"] = event_attrs["exception.type"]
                    if "exception.message" in event_attrs:
                        chain_info["error_message"] = event_attrs["exception.message"]
                    break

        return chain_info

    def _extract_agent_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent interaction details from an AGENT span."""
        attributes = span.get("attributes", {})
        status = span.get("status", {})

        agent_info = {
            "type": "agent",
            "name": span.get("name", "Unknown agent"),
            "status": "success" if status.get("status_code") == "OK" else "error",
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

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> List[Dict]:
        """Extract LLM calls and tool calls from SmoL Agents telemetry."""
        calls = []

        for span in telemetry:
            calls.append(self.extract_interaction(span)[1])
        return calls

    def extract_interaction(self, span: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract interaction details from a span."""
        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        if span_kind == "LLM" or "LiteLLMModel.__call__" in span.get("name", ""):
            return "LLM", self._extract_llm_interaction(span)
        elif span_kind == "CHAIN":
            return "CHAIN", self._extract_chain_interaction(span)
        elif span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        elif "tool.name" in attributes or span.get("name", "").startswith("SimpleTool"):
            return "TOOL", self._extract_tool_interaction(span)
        else:
            raise ValueError(
                f"Unknown span kind: {span_kind} or unsupported span name: {span.get('name', '')}"
            )
