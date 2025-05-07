from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from litellm.cost_calculator import cost_per_token
from pydantic import BaseModel, ConfigDict, Field

from any_agent.config import AgentFramework
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


class CountInfo(BaseModel):
    """Token Count information."""

    token_count_prompt: int
    token_count_completion: int

    model_config = ConfigDict(extra="forbid")


class CostInfo(BaseModel):
    """Cost information."""

    cost_prompt: float
    cost_completion: float

    model_config = ConfigDict(extra="forbid")


class TotalTokenUseAndCost(BaseModel):
    """Total token use and cost information."""

    total_token_count_prompt: int
    total_token_count_completion: int
    total_cost_prompt: float
    total_cost_completion: float

    total_cost: float
    total_tokens: int

    model_config = ConfigDict(extra="forbid")


def extract_token_use_and_cost(
    attributes: Mapping[str, AttributeValue],
) -> CostInfo | None:
    """Use litellm and openinference keys to extract token use and cost."""
    span_info: dict[str, AttributeValue] = {}

    for key in ["llm.token_count.prompt", "llm.token_count.completion"]:
        if key in attributes:
            name = key.split(".")[-1]
            span_info[f"token_count_{name}"] = attributes[key]

    if not span_info:
        return None

    new_info: dict[str, float] = {}
    try:
        cost_prompt, cost_completion = cost_per_token(
            model=str(attributes.get("llm.model_name", "")),
            prompt_tokens=int(attributes.get("llm.token_count.prompt", 0)),  # type: ignore[arg-type]
            completion_tokens=int(span_info.get("llm.token_count.completion", 0)),  # type: ignore[arg-type]
        )
        new_info["cost_prompt"] = cost_prompt
        new_info["cost_completion"] = cost_completion
    except Exception as e:
        msg = f"Error computing cost_per_token: {e}"
        logger.warning(msg)
        new_info["cost_prompt"] = 0.0
        new_info["cost_completion"] = 0.0

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
        cost_info = extract_token_use_and_cost(self.attributes)
        if cost_info:
            self.set_attributes(cost_info.model_dump(exclude_none=True))

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set attributes for the span."""
        for key, value in attributes.items():
            if key in self.attributes:
                logger.warning("Overwriting attribute %s with %s", key, value)
            self.attributes[key] = value


class AgentTrace(BaseModel):
    """A trace that can be exported to JSON or printed to the console."""

    spans: list[AgentSpan] = Field(default_factory=list)
    """A list of [`AgentSpan`][any_agent.tracing.trace.AgentSpan] that form the trace.
    """

    final_output: str | None = None
    """Contains the final output message returned by the agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_total_cost(self) -> TotalTokenUseAndCost:
        """Return the current total cost and token usage statistics."""
        counts: list[CountInfo] = []
        costs: list[CostInfo] = []
        for span in self.spans:
            if span.attributes and "cost_prompt" in span.attributes:
                count = CountInfo(
                    token_count_prompt=span.attributes["llm.token_count.prompt"],
                    token_count_completion=span.attributes[
                        "llm.token_count.completion"
                    ],
                )
                cost = CostInfo(
                    cost_prompt=span.attributes["cost_prompt"],
                    cost_completion=span.attributes["cost_completion"],
                )
                counts.append(count)
                costs.append(cost)

        total_cost = sum(cost.cost_prompt + cost.cost_completion for cost in costs)
        total_tokens = sum(
            count.token_count_prompt + count.token_count_completion for count in counts
        )
        total_token_count_prompt = sum(count.token_count_prompt for count in counts)
        total_token_count_completion = sum(
            count.token_count_completion for count in counts
        )
        total_cost_prompt = sum(cost.cost_prompt for cost in costs)
        total_cost_completion = sum(cost.cost_completion for cost in costs)
        return TotalTokenUseAndCost(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_token_count_prompt=total_token_count_prompt,
            total_token_count_completion=total_token_count_completion,
            total_cost_prompt=total_cost_prompt,
            total_cost_completion=total_cost_completion,
        )


def _is_tracing_supported(agent_framework: AgentFramework) -> bool:
    """Check if tracing is supported for the given agent framework."""
    # Agno not yet supported https://github.com/Arize-ai/openinference/issues/1302
    # Google ADK not yet supported https://github.com/Arize-ai/openinference/issues/1506
    if agent_framework in (
        AgentFramework.AGNO,
        AgentFramework.GOOGLE,
        AgentFramework.TINYAGENT,
    ):
        return False
    return True
