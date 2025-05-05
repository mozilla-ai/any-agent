from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, assert_never

from any_agent import AgentFramework
from any_agent.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from any_agent.tracing.trace import AgentSpan, AgentTrace


class TracingProcessor(ABC):
    """Base class for processing tracing data from different agent types."""

    MAX_EVIDENCE_LENGTH: ClassVar[int] = 400

    @classmethod
    def create(
        cls, agent_framework_raw: AgentFramework | str
    ) -> TracingProcessor | None:
        """Create the appropriate tracing processor."""
        agent_framework = AgentFramework.from_string(agent_framework_raw)

        if agent_framework is AgentFramework.LANGCHAIN:
            from any_agent.tracing.processors.langchain import (
                LangchainTracingProcessor,
            )

            return LangchainTracingProcessor()
        if agent_framework is AgentFramework.SMOLAGENTS:
            from any_agent.tracing.processors.smolagents import (
                SmolagentsTracingProcessor,
            )

            return SmolagentsTracingProcessor()
        if agent_framework is AgentFramework.OPENAI:
            from any_agent.tracing.processors.openai import (
                OpenAITracingProcessor,
            )

            return OpenAITracingProcessor()
        if agent_framework is AgentFramework.LLAMA_INDEX:
            from any_agent.tracing.processors.llama_index import (
                LlamaIndexTracingProcessor,
            )

            return LlamaIndexTracingProcessor()

        if (
            agent_framework is AgentFramework.GOOGLE
            or agent_framework is AgentFramework.AGNO
            or agent_framework is AgentFramework.TINYAGENT
        ):
            return None

        assert_never(agent_framework)

    @abstractmethod
    def _extract_hypothesis_answer(self, trace: AgentTrace) -> str:
        """Extract the hypothesis agent final answer from the trace."""

    @abstractmethod
    def _get_agent_framework(self) -> AgentFramework:
        """Get the agent type associated with this processor."""

    @abstractmethod
    def _extract_llm_interaction(self, span: AgentSpan) -> Mapping[str, Any]:
        """Extract interaction details of a span of type LLM."""

    @abstractmethod
    def _extract_tool_interaction(self, span: AgentSpan) -> Mapping[str, Any]:
        """Extract interaction details of a span of type TOOL."""

    @abstractmethod
    def _extract_chain_interaction(self, span: AgentSpan) -> Mapping[str, Any]:
        """Extract interaction details of a span of type CHAIN."""

    @abstractmethod
    def _extract_agent_interaction(self, span: AgentSpan) -> Mapping[str, Any]:
        """Extract interaction details of a span of type AGENT."""

    def extract_evidence(self, trace: AgentTrace) -> str:
        """Extract relevant evidence."""
        calls = self._extract_trace_data(trace)
        return self._format_evidence(calls)

    def _format_evidence(self, calls: Sequence[Mapping[str, Any]]) -> str:
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
        """Parse a string that has items of a dict with key-value pairs separated by '='.

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

    def _extract_trace_data(
        self,
        trace: AgentTrace,
    ) -> list[Mapping[str, Any]]:
        """Extract the agent-specific data from trace."""
        calls = []

        for span in trace.spans:
            calls.append(self.extract_interaction(span)[1])

        return calls

    def extract_interaction(
        self,
        span: AgentSpan,
    ) -> tuple[str, Mapping[str, Any]]:
        """Extract interaction details from a span."""
        span_kind = span.attributes.get("openinference.span.kind", "")

        if span_kind == "LLM" or "LiteLLMModel.__call__" in span.name:
            return "LLM", self._extract_llm_interaction(span)
        if "tool.name" in span.attributes or span.name.endswith("Tool"):
            return "TOOL", self._extract_tool_interaction(span)
        if span_kind == "CHAIN":
            return "CHAIN", self._extract_chain_interaction(span)
        if span_kind == "AGENT":
            return "AGENT", self._extract_agent_interaction(span)
        logger.warning(f"Unknown span kind: {span_kind}. Span: {span}")
        return "UNKNOWN", {}
