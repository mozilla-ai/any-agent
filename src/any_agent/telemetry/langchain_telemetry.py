from typing import Any, Dict, List
import json
from any_agent import AgentFramework
from langchain_core.messages import BaseMessage


from any_agent.logging import logger
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

    def _extract_llm_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract LLM interaction details from a span."""
        attributes = span.get("attributes", {})
        span_info = {
            "model": attributes.get("llm.model_name", "Unknown model"),
            "type": "reasoning",
        }

        if "llm.input_messages.0.message.content" in attributes:
            span_info["input"] = attributes["llm.input_messages.0.message.content"]

        if "llm.output_messages.0.message.content" in attributes:
            span_info["output"] = attributes["llm.output_messages.0.message.content"]

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

        if "input.value" in attributes:
            try:
                input_value = json.loads(attributes["input.value"])
                tool_info["input"] = input_value
            except Exception:
                tool_info["input"] = attributes["input.value"]

        if "output.value" in attributes:
            try:
                output_value = json.loads(attributes["output.value"])
                if "output" in output_value:
                    parsed_output = self.parse_generic_key_value_string(
                        output_value["output"]
                    )
                    if "content" in parsed_output:
                        tool_info["output"] = parsed_output["content"]
                    else:
                        tool_info["output"] = parsed_output
                else:
                    tool_info["output"] = output_value
            except Exception:
                tool_info["output"] = attributes["output.value"]

        return tool_info

    def _extract_chain_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chain interaction details from a span."""
        attributes = span.get("attributes", {})
        chain_info = {
            "type": "chain",
            "name": span.get("name", "Unknown chain"),
        }

        # Extract input from the chain
        if "input.value" in attributes:
            try:
                input_data = json.loads(attributes["input.value"])
                if "messages" in input_data and isinstance(
                    input_data["messages"], list
                ):
                    # Extract message content if available
                    messages = []
                    for msg in input_data["messages"]:
                        if isinstance(msg, list) and len(msg) > 1:
                            # Format: ["role", "content"]
                            messages.append({"role": msg[0], "content": msg[1]})
                    if messages:
                        chain_info["input"] = messages
                    else:
                        chain_info["input"] = input_data
                else:
                    chain_info["input"] = input_data
            except Exception:
                chain_info["input"] = attributes["input.value"]

        # Extract output from the chain
        if "output.value" in attributes:
            try:
                output_data = json.loads(attributes["output.value"])
                if "messages" in output_data:
                    # Try to parse the messages
                    parsed_messages = []
                    for msg in output_data["messages"]:
                        parsed_msg = self.parse_generic_key_value_string(msg)
                        if parsed_msg:
                            parsed_messages.append(parsed_msg)

                    if parsed_messages:
                        chain_info["output"] = parsed_messages
                    else:
                        chain_info["output"] = output_data["messages"]
                else:
                    chain_info["output"] = output_data
            except Exception:
                chain_info["output"] = attributes["output.value"]

        return chain_info

    def _extract_agent_interaction(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent interaction details from a span."""
        attributes = span.get("attributes", {})
        agent_info = {
            "type": "agent",
            "name": span.get("name", "Agent"),
        }

        # Extract input from the agent span
        if "input.value" in attributes:
            try:
                input_data = json.loads(attributes["input.value"])
                if "messages" in input_data and isinstance(
                    input_data["messages"], list
                ):
                    if len(input_data["messages"]) > 0:
                        # For Langchain agents, messages might be serialized as strings
                        message = self.parse_generic_key_value_string(
                            input_data["messages"][0]
                        )
                        if message and "content" in message:
                            agent_info["input"] = message["content"]
                        else:
                            agent_info["input"] = input_data["messages"][0]
                    else:
                        agent_info["input"] = input_data
                else:
                    agent_info["input"] = input_data
            except Exception:
                agent_info["input"] = attributes["input.value"]

        # Extract direct LLM input if available
        if "llm.input_messages.0.message.content" in attributes:
            agent_info["query"] = attributes["llm.input_messages.0.message.content"]

        # Extract output from the agent span
        if "output.value" in attributes:
            try:
                output_data = json.loads(attributes["output.value"])
                if "messages" in output_data and isinstance(
                    output_data["messages"], list
                ):
                    if len(output_data["messages"]) > 0:
                        # For Langchain agents, messages might be serialized as strings
                        message = self.parse_generic_key_value_string(
                            output_data["messages"][0]
                        )
                        if message and "content" in message:
                            agent_info["output"] = message["content"]
                        else:
                            agent_info["output"] = output_data["messages"][0]
                    else:
                        agent_info["output"] = output_data
                else:
                    agent_info["output"] = output_data
            except Exception:
                agent_info["output"] = attributes["output.value"]

        # Extract metadata if available
        if "metadata" in attributes:
            try:
                metadata = json.loads(attributes["metadata"])
                agent_info["metadata"] = metadata
            except Exception:
                agent_info["metadata"] = attributes["metadata"]

        return agent_info

    def _extract_telemetry_data(self, telemetry: List[Dict[str, Any]]) -> List[Dict]:
        """Extract LLM calls and tool calls from LangChain telemetry."""
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
        elif "tool.name" in attributes or span.get("name", "").endswith("Tool"):
            return "TOOL", self._extract_tool_interaction(span)
        elif span_kind == "CHAIN":
            return "CHAIN", self._extract_chain_interaction(span)
        elif span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        else:
            logger.warning(f"Unknown span kind: {span_kind}. Span: {span}")
            return "UNKNOWN", {}
