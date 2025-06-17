# mypy: disable-error-code="arg-type,attr-defined"
from __future__ import annotations

import json
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from litellm.cost_calculator import cost_per_token
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from any_agent.logging import logger

from .otel_types import (
    AttributeValue,
    Event,
    Link,
    Resource,
    SpanContext,
    SpanKind,
    Status,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span


class TokenInfo(BaseModel):
    """Token Count information."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total number of tokens."""
        return self.input_tokens + self.output_tokens

    model_config = ConfigDict(extra="forbid")


class CostInfo(BaseModel):
    """Cost information."""

    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        """Total cost."""
        return self.input_cost + self.output_cost

    model_config = ConfigDict(extra="forbid")


def compute_cost_info(
    attributes: Mapping[str, AttributeValue],
) -> CostInfo | None:
    """Use litellm to compute cost."""
    if not any(
        key in attributes
        for key in ["gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"]
    ):
        return None

    new_info: dict[str, float] = {}
    try:
        cost_prompt, cost_completion = cost_per_token(
            model=str(attributes.get("gen_ai.request.model", "")),
            prompt_tokens=int(attributes.get("gen_ai.usage.input_tokens", 0)),  # type: ignore[arg-type]
            completion_tokens=int(attributes.get("gen_ai.usage.output_tokens", 0)),  # type: ignore[arg-type]
        )
        new_info["input_cost"] = cost_prompt
        new_info["output_cost"] = cost_completion
    except Exception as e:
        msg = f"Error computing cost_per_token: {e}"
        logger.warning(msg)
        new_info["input_cost"] = 0.0
        new_info["output_cost"] = 0.0
    return CostInfo.model_validate(new_info)


class AgentMessage(BaseModel):
    """A message that can be exported to JSON or printed to the console."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str

    model_config = ConfigDict(extra="forbid")


class AgentSpan(BaseModel):
    """A span that can be exported to JSON or printed to the console."""

    name: str
    kind: SpanKind
    parent: SpanContext | None = None
    start_time: int | None = None
    end_time: int | None = None
    status: Status
    context: SpanContext
    attributes: dict[str, Any]
    links: list[Link]
    events: list[Event]
    resource: Resource

    model_config = ConfigDict(arbitrary_types_allowed=False)

    @classmethod
    def from_otel(cls, otel_span: Span) -> AgentSpan:
        """Create an AgentSpan from an OTEL Span."""
        return cls(
            name=otel_span.name,
            kind=SpanKind.from_otel(otel_span.kind),
            parent=SpanContext.from_otel(otel_span.parent),
            start_time=otel_span.start_time,
            end_time=otel_span.end_time,
            status=Status.from_otel(otel_span.status),
            context=SpanContext.from_otel(otel_span.context),
            attributes=dict(otel_span.attributes) if otel_span.attributes else {},
            links=[Link.from_otel(link) for link in otel_span.links],
            events=[Event.from_otel(event) for event in otel_span.events],
            resource=Resource.from_otel(otel_span.resource),
        )

    def to_readable_span(self) -> ReadableSpan:
        """Create an ReadableSpan from the AgentSpan."""
        return ReadableSpan(
            name=self.name,
            kind=self.kind,
            parent=self.parent,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status,
            context=self.context,
            attributes=self.attributes,
            links=self.links,
            events=self.events,
            resource=self.resource,
        )

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set attributes for the span."""
        for key, value in attributes.items():
            if key in self.attributes:
                logger.warning("Overwriting attribute %s with %s", key, value)
            self.attributes[key] = value

    def is_agent_invocation(self) -> bool:
        """Check whether this span is an agent invocation (the very first span)."""
        return self.attributes.get("gen_ai.operation.name") == "invoke_agent"

    def is_llm_call(self) -> bool:
        """Check whether this span is a call to an LLM."""
        return self.attributes.get("gen_ai.operation.name") == "call_llm"

    def is_tool_execution(self) -> bool:
        """Check whether this span is an execution of a tool."""
        return self.attributes.get("gen_ai.operation.name") == "execute_tool"

    def get_input_messages(self) -> list[AgentMessage] | None:
        """Extract input messages from an LLM call span.

        Returns:
            List of message dicts with 'role' and 'content' keys, or None if not available.

        """
        if not self.is_llm_call():
            msg = "Span is not an LLM call"
            raise ValueError(msg)

        messages_json = self.attributes.get("gen_ai.input.messages")
        if not messages_json:
            logger.debug("No input messages found in span")
            return None

        try:
            parsed_messages = json.loads(messages_json)
            # Ensure it's a list of dicts
        except (json.JSONDecodeError, TypeError) as e:
            msg = "Failed to parse input messages from span"
            logger.error(msg)
            raise ValueError(msg) from e
        if not isinstance(parsed_messages, list):
            msg = "Input messages are not a list of messages"
            raise ValueError(msg)
        return [AgentMessage.model_validate(msg) for msg in parsed_messages]

    def get_output_content(self) -> str | None:
        """Extract output content from an LLM call or tool execution span.

        Returns:
            The output content as a string, or None if not available.

        """
        if not self.is_llm_call() and not self.is_tool_execution():
            msg = "Span is not an LLM call or tool execution"
            raise ValueError(msg)

        output = self.attributes.get("gen_ai.output")
        if not output:
            logger.debug("No output found in span")
            return None
        return str(output)


class AgentTrace(BaseModel):
    """A trace that can be exported to JSON or printed to the console."""

    spans: list[AgentSpan] = Field(default_factory=list)
    """A list of [`AgentSpan`][any_agent.tracing.agent_trace.AgentSpan] that form the trace.
    """

    final_output: str | BaseModel | None = Field(default=None)
    """Contains the final output message returned by the agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("final_output")
    def serialize_final_output(self, value: str | BaseModel | None) -> Any:
        """Serialize the final_output and handle any BaseModel subclass."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, BaseModel):
            # This will properly serialize any BaseModel subclass
            return value.model_dump()
        return value

    def _invalidate_tokens_and_cost_cache(self) -> None:
        """Clear the cached tokens_and_cost property if it exists."""
        if "tokens" in self.__dict__:
            del self.tokens
        if "cost" in self.__dict__:
            del self.cost

    def add_span(self, span: AgentSpan | Span) -> None:
        """Add an AgentSpan to the trace and clear the tokens_and_cost cache if present."""
        if not isinstance(span, AgentSpan):
            span = AgentSpan.from_otel(span)
        self.spans.append(span)
        self._invalidate_tokens_and_cost_cache()

    def add_spans(self, spans: list[AgentSpan]) -> None:
        """Add a list of AgentSpans to the trace and clear the tokens_and_cost cache if present."""
        self.spans.extend(spans)
        self._invalidate_tokens_and_cost_cache()

    def spans_to_messages(self) -> list[AgentMessage]:
        """Convert spans to standard message format.

        Returns:
            List of message dicts with 'role' and 'content' keys.

        """
        messages: list[AgentMessage] = []

        # Process spans in chronological order (excluding the final invoke_agent span)
        # Filter out any agent invocation spans
        filtered_spans: list[AgentSpan] = []
        for span in self.spans:
            if not span.is_agent_invocation():
                filtered_spans.append(span)

        for span in filtered_spans:
            if span.is_llm_call():
                # Extract input messages from the span
                input_messages = span.get_input_messages()
                if input_messages:
                    for msg in input_messages:
                        if not any(
                            existing.role == msg.role
                            and existing.content == msg.content
                            for existing in messages
                        ):
                            messages.append(msg)

                # Add the assistant's response
                output_content = span.get_output_content()
                if output_content:
                    # Avoid duplicate assistant messages
                    if not (
                        messages
                        and messages[-1].role == "assistant"
                        and messages[-1].content == output_content
                    ):
                        messages.append(
                            AgentMessage(role="assistant", content=output_content)
                        )

            elif span.is_tool_execution():
                # For tool executions, include the result in the conversation
                output_content = span.get_output_content()
                if output_content:
                    tool_name = span.attributes["gen_ai.tool.name"]
                    tool_args = span.attributes["gen_ai.tool.args"]
                    messages.append(
                        AgentMessage(
                            role="assistant",
                            content=f"[Tool {tool_name} executed: {output_content} with args: {tool_args}]",
                        )
                    )

        return messages

    @property
    def duration(self) -> timedelta:
        """Duration of the parent `invoke_agent` span as a datetime.timedelta object.

        The duration is computed from the span's start and end time (in nanoseconds).

        Raises ValueError if:
            - There are no spans.
            - The invoke_agent span is not the last span.
            - Any of the start/end times are missing.
        """
        if not self.spans:
            msg = "No spans found in trace"
            raise ValueError(msg)
        span = self.spans[-1]
        if not span.is_agent_invocation():
            msg = "Last span is not `invoke_agent`"
            raise ValueError(msg)
        if span.start_time is not None and span.end_time is not None:
            duration_ns = span.end_time - span.start_time
            return timedelta(seconds=duration_ns / 1_000_000_000)
        msg = "Start or end time is missing for the `invoke_agent` span"
        raise ValueError(msg)

    @cached_property
    def tokens(self) -> TokenInfo:
        """The current total token count for this trace. Cached after first computation."""
        sum_input_tokens = 0
        sum_output_tokens = 0
        for span in self.spans:
            if span.is_llm_call():
                sum_input_tokens += span.attributes.get("gen_ai.usage.input_tokens", 0)
                sum_output_tokens += span.attributes.get(
                    "gen_ai.usage.output_tokens", 0
                )
        return TokenInfo(input_tokens=sum_input_tokens, output_tokens=sum_output_tokens)

    @cached_property
    def cost(self) -> CostInfo:
        """The current total cost for this trace. Cached after first computation."""
        sum_input_cost = 0.0
        sum_output_cost = 0.0
        for span in self.spans:
            if span.is_llm_call():
                cost_info = compute_cost_info(span.attributes)
                if cost_info:
                    sum_input_cost += cost_info.input_cost
                    sum_output_cost += cost_info.output_cost
        return CostInfo(input_cost=sum_input_cost, output_cost=sum_output_cost)
