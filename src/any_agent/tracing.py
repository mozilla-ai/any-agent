import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, assert_never

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.config import AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.telemetry import (
    TelemetryProcessor,
    TokenUseAndCost,
    TotalTokenUseAndCost,
    extract_token_use_and_cost,
)
from any_agent.telemetry.types import (
    AttributeValue,
    Link,
    SpanContext,
    SpanKind,
    Status,
)
from any_agent.telemetry.types import (
    Event as EventModel,
)
from any_agent.telemetry.types import (
    Resource as ResourceModel,
)


# only keep a few things that we care about from the AnyAgentSpan,
# but move it to this class because otherwise we can't recreate it
class AnyAgentSpan(BaseModel):
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
    events: list[EventModel]
    resource: ResourceModel

    model_config = ConfigDict(arbitrary_types_allowed=False)

    @classmethod
    def from_readable_span(cls, readable_span: ReadableSpan) -> "AnyAgentSpan":
        """Create an AnyAgentSpan from a ReadableSpan."""
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
            events=[EventModel.from_otel(event) for event in readable_span.events],
            resource=ResourceModel.from_otel(readable_span.resource),
        )

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set attributes for the span."""
        for key, value in attributes.items():
            if key in self.attributes:
                logger.warning("Overwriting attribute %s with %s", key, value)
            self.attributes[key] = value

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Set a single attribute for the span."""
        return self.set_attributes({key: value})


class AnyAgentTrace(BaseModel):
    """A trace that can be exported to JSON or printed to the console."""

    spans: list[AnyAgentSpan]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_total_cost(self) -> TotalTokenUseAndCost:
        """Return the current total cost and token usage statistics."""
        costs: list[TokenUseAndCost] = []
        for span in self.spans:
            if span.attributes and "cost_prompt" in span.attributes:
                token_use_and_cost = TokenUseAndCost(
                    token_count_prompt=span.attributes["llm.token_count.prompt"],
                    token_count_completion=span.attributes[
                        "llm.token_count.completion"
                    ],
                    cost_prompt=span.attributes["cost_prompt"],
                    cost_completion=span.attributes["cost_completion"],
                )
                costs.append(token_use_and_cost)

        total_cost = sum(cost.cost_prompt + cost.cost_completion for cost in costs)
        total_tokens = sum(
            cost.token_count_prompt + cost.token_count_completion for cost in costs
        )
        total_token_count_prompt = sum(cost.token_count_prompt for cost in costs)
        total_token_count_completion = sum(
            cost.token_count_completion for cost in costs
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


class TraceSpanExporter(SpanExporter):  # noqa: D101
    def __init__(
        self,
        agent_framework: AgentFramework,
        tracing_config: TracingConfig,
        file_name: str,
        tracer_trace: AnyAgentTrace,
    ):
        """Initialize the JsonFileSpanExporter."""
        self.tracer_trace: AnyAgentTrace = tracer_trace  # so that this exporter can set the trace and can be used to get the trace
        self.processor = TelemetryProcessor.create(agent_framework)
        self.file_name = file_name
        self.tracing_config = tracing_config
        self.save: bool = tracing_config.save
        if self.save and not os.path.exists(self.tracing_config.output_dir):
            os.makedirs(self.tracing_config.output_dir)
        if self.save and not os.path.exists(self.file_name):
            with open(self.file_name, "w", encoding="utf-8") as f:
                json.dump([], f)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: D102
        # Add new spans
        new_spans = []
        for readable_span in spans:
            span = AnyAgentSpan.from_readable_span(readable_span)
            try:
                # Try to parse the span data from to_json() if it returns a string
                span_kind, _ = self.processor.extract_interaction(span)
                if span_kind == "LLM" and self.tracing_config.cost_info:
                    cost_info = extract_token_use_and_cost(span.attributes)
                    span.set_attributes(cost_info.model_dump())

            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning("Failed to parse span data, %s, %s", span, e)
                continue
            new_spans.append(span)
        self.tracer_trace.spans.extend(new_spans)
        with open(self.file_name, "w", encoding="utf-8") as f:
            json.dump(
                [span.model_dump_json() for span in self.tracer_trace.spans],
                f,
                indent=2,
            )

        return SpanExportResult.SUCCESS


class RichConsoleSpanExporter(SpanExporter):  # noqa: D101
    def __init__(self, agent_framework: AgentFramework, tracing_config: TracingConfig):  # noqa: D107
        self.processor = TelemetryProcessor.create(agent_framework)
        self.console = Console()
        self.tracing_config = tracing_config

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: D102
        for readable_span in spans:
            style = None
            span = AnyAgentSpan.from_readable_span(readable_span)
            try:
                span_kind, interaction = self.processor.extract_interaction(span)

                style = getattr(self.tracing_config, span_kind.lower(), None)

                if not style or interaction == {}:
                    continue

                self.console.rule(
                    span_kind,
                    style=style,
                )
                for key, value in interaction.items():
                    if key == "output":
                        self.console.print(
                            Panel(
                                Markdown(str(value or "")),
                                title="Output",
                            ),
                        )
                    else:
                        self.console.print(f"{key}: {value}")

                if span_kind == "LLM" and self.tracing_config.cost_info:
                    cost_info = extract_token_use_and_cost(span.attributes)
                    for key, value in cost_info.model_dump().items():
                        self.console.print(f"{key}: {value}")

            except Exception:
                self.console.print_exception()
            if style:
                self.console.rule(style=style)
        return SpanExportResult.SUCCESS


class Instrumenter(Protocol):  # noqa: D101
    def instrument(self, *, tracer_provider: TracerProvider) -> None: ...  # noqa: D102

    def uninstrument(self) -> None: ...  # noqa: D102


def _get_instrumenter_by_framework(framework: AgentFramework) -> Instrumenter:
    if framework is AgentFramework.OPENAI:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        return OpenAIAgentsInstrumentor()

    if framework is AgentFramework.SMOLAGENTS:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        return SmolagentsInstrumentor()

    if framework is AgentFramework.LANGCHAIN:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        return LangChainInstrumentor()

    if framework is AgentFramework.LLAMA_INDEX:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        return LlamaIndexInstrumentor()

    if (
        framework is AgentFramework.GOOGLE
        or framework is AgentFramework.AGNO
        or framework is AgentFramework.TINYAGENT
    ):
        msg = f"{framework} tracing is not supported."
        raise NotImplementedError(msg)

    assert_never(framework)


class Tracer:
    """Tracer is responsible for managing all things tracing for an agent."""

    def __init__(
        self,
        agent_framework: AgentFramework,
        tracing_config: TracingConfig,
    ):
        """Initialize the Tracer and set up tracing filepath, if enabled."""
        self.agent_framework = agent_framework
        # Set it to None at first To avoid AttributeError on __del__ if _get_instrumenter_by_framework throws exception
        self.instrumenter: Instrumenter | None = None
        self.instrumenter = _get_instrumenter_by_framework(
            agent_framework
        )  # Fail fast if framework is not supported
        self.tracing_config = tracing_config
        self.trace_filepath: str | None = None
        self.trace: AnyAgentTrace = AnyAgentTrace(spans=[])

        self._setup_tracing()

    def uninstrument(self) -> None:
        """Uninstrument the tracer."""
        if self.instrumenter:
            self.instrumenter.uninstrument()
            self.instrumenter = None

    def __del__(self) -> None:
        """Stop the openinference instrumentation when the tracer is deleted."""
        self.uninstrument()

    def _setup_tracing(self) -> None:
        """Set up tracing for the agent."""
        tracer_provider = TracerProvider()

        if not self.instrumenter:
            msg = "Instrumenter not found for the agent framework."
            raise ValueError(msg)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.trace_filepath = f"{self.tracing_config.output_dir}/{self.agent_framework.name}-{timestamp}.json"
        json_file_exporter = TraceSpanExporter(
            agent_framework=self.agent_framework,
            tracing_config=self.tracing_config,
            file_name=self.trace_filepath,
            tracer_trace=self.trace,
        )
        span_processor = SimpleSpanProcessor(json_file_exporter)
        tracer_provider.add_span_processor(span_processor)

        if self.tracing_config.console:
            processor = BatchSpanProcessor(
                RichConsoleSpanExporter(self.agent_framework, self.tracing_config),
            )
            tracer_provider.add_span_processor(processor)

        trace.set_tracer_provider(tracer_provider)

        self.instrumenter.instrument(tracer_provider=tracer_provider)

    @property
    def is_enabled(self) -> bool:
        """Whether tracing is enabled."""
        return self.tracing_config.save or self.tracing_config.console

    def get_trace(self) -> AnyAgentTrace | None:
        """Return the trace data if file tracing is enabled."""
        if self.trace:
            return self.trace
        msg = "Tracing is not enabled."
        logger.warning(msg)
        return None
