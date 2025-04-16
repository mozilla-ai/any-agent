import json
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from any_agent import AgentFramework
from any_agent.logging import logger


class TelemetryProcessor(ABC):
    """Base class for processing telemetry data from different agent types."""

    MAX_EVIDENCE_LENGTH: ClassVar[int] = 400

    @classmethod
    def create(cls, agent_framework: AgentFramework) -> "TelemetryProcessor":
        """Factory method to create the appropriate telemetry processor."""
        if agent_framework == AgentFramework.LANGCHAIN:
            from any_agent.telemetry.langchain_telemetry import (
                LangchainTelemetryProcessor,
            )

            return LangchainTelemetryProcessor()
        if agent_framework == AgentFramework.SMOLAGENTS:
            from any_agent.telemetry.smolagents_telemetry import (
                SmolagentsTelemetryProcessor,
            )

            return SmolagentsTelemetryProcessor()
        if agent_framework == AgentFramework.OPENAI:
            from any_agent.telemetry.openai_telemetry import (
                OpenAITelemetryProcessor,
            )

            return OpenAITelemetryProcessor()
        if agent_framework == AgentFramework.LLAMAINDEX:
            from any_agent.telemetry.llama_index_telemetry import (
                LlamaIndexTelemetryProcessor,
            )

            return LlamaIndexTelemetryProcessor()
        msg = f"Unsupported agent type {agent_framework}"
        raise ValueError(msg)

    @abstractmethod
    def extract_hypothesis_answer(self, trace: list[dict[str, Any]]) -> str:
        """Extract the hypothesis agent final answer from the trace."""

    @abstractmethod
    def _get_agent_framework(self) -> AgentFramework:
        """Get the agent type associated with this processor."""

    @abstractmethod
    def _extract_llm_interaction(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract interaction details of a span of type LLM"""

    @abstractmethod
    def _extract_tool_interaction(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract interaction details of a span of type TOOL"""

    @abstractmethod
    def _extract_chain_interaction(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract interaction details of a span of type CHAIN"""

    @abstractmethod
    def _extract_agent_interaction(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract interaction details of a span of type AGENT"""

    @staticmethod
    def determine_agent_framework(trace: list[dict[str, Any]]) -> AgentFramework:
        """Determine the agent type based on the trace.
        These are not really stable ways to find it, because we're waiting on some
        reliable method for determining the agent type. This is a temporary solution.
        """
        for span in trace:
            if "langchain" in span.get("attributes", {}).get("input.value", ""):
                logger.info("Agent type is LANGCHAIN")
                return AgentFramework.LANGCHAIN
            if span.get("attributes", {}).get("smolagents.max_steps"):
                logger.info("Agent type is SMOLAGENTS")
                return AgentFramework.SMOLAGENTS
            # This is extremely fragile but there currently isn't
            # any specific key to indicate the agent type
            if span.get("name") == "response":
                logger.info("Agent type is OPENAI")
                return AgentFramework.OPENAI
        msg = "Could not determine agent type from trace, or agent type not supported"
        raise ValueError(msg)

    def extract_evidence(self, telemetry: list[dict[str, Any]]) -> str:
        """Extract relevant telemetry evidence."""
        calls = self._extract_telemetry_data(telemetry)
        return self._format_evidence(calls)

    def _format_evidence(self, calls: list[dict]) -> str:
        """Format extracted data into a standardized output format."""
        evidence = f"## {self._get_agent_framework().name} Agent Execution\n\n"

        for idx, call in enumerate(calls, start=1):
            evidence += f"### Call {idx}\n"

            # Truncate any values that are too long
            call = {
                k: (
                    v[: self.MAX_EVIDENCE_LENGTH] + "..."
                    if isinstance(v, str) and len(v) > self.MAX_EVIDENCE_LENGTH
                    else v
                )
                for k, v in call.items()
            }

            # Use ensure_ascii=False to prevent escaping Unicode characters
            evidence += json.dumps(call, indent=2, ensure_ascii=False) + "\n\n"

        return evidence

    @staticmethod
    def parse_generic_key_value_string(text: str) -> dict[str, str]:
        """
        Parse a string that has items of a dict with key-value pairs separated by '='.
        Only splits on '=' signs, handling quoted strings properly.
        """
        pattern = r"(\w+)=('.*?'|\".*?\"|[^'\"=]*?)(?=\s+\w+=|\s*$)"
        result = {}

        matches = re.findall(pattern, text)
        for key, value in matches:
            # Clean up the key
            key = key.strip()

            # Clean up the value - remove surrounding quotes if present
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]

            # Store in result dictionary
            result[key] = value

        return result

    def _extract_telemetry_data(self, telemetry: list[dict[str, Any]]) -> list[dict]:
        """Extract LLM calls and tool calls from LangChain telemetry."""
        calls = []

        for span in telemetry:
            calls.append(self.extract_interaction(span)[1])

        return calls

    def extract_interaction(self, span: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract interaction details from a span."""
        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        if span_kind == "LLM" or "LiteLLMModel.__call__" in span.get("name", ""):
            return "LLM", self._extract_llm_interaction(span)
        if "tool.name" in attributes or span.get("name", "").endswith("Tool"):
            return "TOOL", self._extract_tool_interaction(span)
        if span_kind == "CHAIN":
            return "CHAIN", self._extract_chain_interaction(span)
        if span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        logger.warning(f"Unknown span kind: {span_kind}. Span: {span}")
        return "UNKNOWN", {}
