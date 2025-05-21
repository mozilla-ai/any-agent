from collections.abc import Mapping
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any

from litellm.cost_calculator import cost_per_token
from pydantic import BaseModel, ConfigDict, Field

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
    from opentelemetry.sdk.trace import ReadableSpan


class TokenInfo(BaseModel):
    """Token Count information."""

    input_tokens: int
    output_tokens: int

    model_config = ConfigDict(extra="forbid")


class CostInfo(BaseModel):
    """Cost information."""

    input_cost: float
    output_cost: float

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
    def from_readable_span(cls, readable_span: "ReadableSpan") -> "AgentSpan":
        """Create an AgentSpan from a ReadableSpan."""
        return cls(
            name=readable_span.name,
            kind=SpanKind.from_otel(readable_span.kind),
            parent=SpanContext.from_otel(readable_span.parent),
            start_time=readable_span.start_time,
            end_time=readable_span.end_time,
            status=Status.from_otel(readable_span.status),
            context=SpanContext.from_otel(readable_span.context),
            attributes=dict(readable_span.attributes)
            if readable_span.attributes
            else {},
            links=[Link.from_otel(link) for link in readable_span.links],
            events=[Event.from_otel(event) for event in readable_span.events],
            resource=Resource.from_otel(readable_span.resource),
        )

    def add_cost_info(self) -> None:
        """Extend attributes with `TokenUseAndCost`."""
        cost_info = compute_cost_info(self.attributes)
        if cost_info:
            self.set_attributes(
                {f"gen_ai.usage.{k}": v for k, v in cost_info.model_dump().items()}
            )

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set attributes for the span."""
        for key, value in attributes.items():
            if key in self.attributes:
                logger.warning("Overwriting attribute %s with %s", key, value)
            self.attributes[key] = value

    def is_llm_call(self):
        """Check whether this span is a call to an LLM."""
        return self.attributes.get("gen_ai.operation.name") == "call_llm"


class AgentTrace(BaseModel):
    """A trace that can be exported to JSON or printed to the console."""

    spans: list[AgentSpan] = Field(default_factory=list)
    """A list of [`AgentSpan`][any_agent.tracing.trace.AgentSpan] that form the trace.
    """

    final_output: str | None = None
    """Contains the final output message returned by the agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _invalidate_usage_and_cost_cache(self) -> None:
        """Clear the cached usage_and_cost property if it exists."""
        if "usage" in self.__dict__:
            del self.usage
        if "cost" in self.__dict__:
            del self.cost

    def add_span(self, span: AgentSpan) -> None:
        """Add an AgentSpan to the trace and clear the usage_and_cost cache if present."""
        self.spans.append(span)
        self._invalidate_usage_and_cost_cache()

    def add_spans(self, spans: list[AgentSpan]) -> None:
        """Add a list of AgentSpans to the trace and clear the usage_and_cost cache if present."""
        self.spans.extend(spans)
        self._invalidate_usage_and_cost_cache()

    @property
    def duration(self) -> timedelta:
        """Returns the duration of the AGENT span named 'any_agent' as a datetime.timedelta object.

        The duration is computed from the span's start and end time (in nanoseconds).
        This functionality relies on the trace behavior where we create a span named 'any_agent' for each agent run,
        which is defined in each agent framework's `run` function.

        Raises ValueError if there are no spans, if the AGENT span is not found, or if start/end times are missing.
        """
        if not self.spans:
            msg = "No spans found in trace"
            raise ValueError(msg)
        for span in self.spans:
            if span.attributes.get("any_agent.run_id"):
                if span.start_time is not None and span.end_time is not None:
                    duration_ns = span.end_time - span.start_time
                    return timedelta(seconds=duration_ns / 1_000_000_000)
                msg = "Start or end time is missing for the AGENT span"
                raise ValueError(msg)
        msg = "Span with any_agent.run_id not found in trace"
        raise ValueError(msg)

    @cached_property
    def usage(self) -> TokenInfo:
        """The current total token usage statistics for this trace. Cached after first computation."""
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
        """The current total cost statistics for this trace. Cached after first computation."""
        sum_input_cost = 0
        sum_output_cost = 0
        for span in self.spans:
            if span.is_llm_call():
                sum_input_cost += span.attributes.get("gen_ai.usage.input_cost", 0)
                sum_output_cost += span.attributes.get("gen_ai.usage.output_cost", 0)
        return CostInfo(input_cost=sum_input_cost, output_cost=sum_output_cost)
