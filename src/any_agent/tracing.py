import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, assert_never

from litellm.cost_calculator import cost_per_token
from opentelemetry import trace
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.util import types
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.config import AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.telemetry import TelemetryProcessor


class TokenUseAndCost(BaseModel):
    """Token use and cost information."""

    token_count_prompt: int
    token_count_completion: int
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
) -> TokenUseAndCost:
    """Use litellm and openinference keys to extract token use and cost."""
    span_info: dict[str, AttributeValue] = {}

    for key in ["llm.token_count.prompt", "llm.token_count.completion"]:
        if key in attributes:
            name = key.split(".")[-1]
            span_info[f"token_count_{name}"] = attributes[key]
    try:
        cost_prompt, cost_completion = cost_per_token(
            model=str(attributes.get("llm.model_name", "")),
            prompt_tokens=int(attributes.get("llm.token_count.prompt", 0)),  # type: ignore[arg-type]
            completion_tokens=int(span_info.get("llm.token_count.completion", 0)),  # type: ignore[arg-type]
        )
        span_info["cost_prompt"] = cost_prompt
        span_info["cost_completion"] = cost_completion
    except Exception as e:
        msg = f"Error computing cost_per_token: {e}"
        logger.warning(msg)
        span_info["cost_prompt"] = 0.0
        span_info["cost_completion"] = 0.0

    return TokenUseAndCost.model_validate(span_info)


# only keep a few things that we care about from the AnyAgentSpan,
# but move it to this class because otherwise we can't recreate it
class AnyAgentSpan(BaseModel):
    """A span that can be exported to JSON or printed to the console."""

    name: str

    kind: trace_api.SpanKind

    parent: trace_api.SpanContext | None

    start_time: int | None

    end_time: int | None

    status: trace_api.Status

    context: trace_api.SpanContext

    attributes: dict[str, types.AttributeValue]

    links: Sequence[trace_api.Link]

    events: Sequence[Event]

    resource: Resource

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_attributes(self, attributes: Mapping[str, types.AttributeValue]) -> None:
        """Set attributes for the span."""
        for key, value in attributes.items():
            if key in self.attributes:
                logger.warning("Overwriting attribute %s with %s", key, value)
            self.attributes[key] = value

    def set_attribute(self, key: str, value: types.AttributeValue) -> None:
        """Set a single attribute for the span."""
        return self.set_attributes({key: value})

    def get_cost_summary(self) -> TotalTokenUseAndCost:
        """Return the current total cost and token usage statistics."""
        costs: list[TokenUseAndCost] = []
        # for span in spans:
        #     pass
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


class JsonFileSpanExporter(SpanExporter):  # noqa: D101
    def __init__(
        self,
        agent_framework: AgentFramework,
        tracing_config: TracingConfig,
        file_name: str,
    ):
        """Initialize the JsonFileSpanExporter."""
        self.processor = TelemetryProcessor.create(agent_framework)
        self.file_name = file_name
        self.tracing_config = tracing_config
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w", encoding="utf-8") as f:
                json.dump([], f)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: D102
        try:
            with open(self.file_name, "r+", encoding="utf-8") as f:
                all_spans: list[AnyAgentSpan] = json.load(f)
                # Add new spans
                for readable_span in spans:
                    # turn the readable span into a generic span that we can update
                    if not readable_span.attributes:
                        msg = "Span must have attributes"
                        raise ValueError(msg)
                    span = AnyAgentSpan(
                        name=readable_span.name,
                        kind=readable_span.kind,
                        context=readable_span.context,
                        parent=readable_span.parent,
                        start_time=readable_span.start_time,
                        end_time=readable_span.end_time,
                        events=readable_span.events,
                        attributes=dict(
                            readable_span.attributes
                        ),  # turn the mapping into a dict so that it's mutable
                        status=readable_span.status,
                        links=readable_span.links,
                        resource=readable_span.resource,
                    )
                    try:
                        # Try to parse the span data from to_json() if it returns a string
                        span_kind, _ = self.processor.extract_interaction(span)
                        if (
                            span_kind == "LLM"
                            and self.tracing_config.cost_info
                            and span.attributes
                        ):
                            cost_info = extract_token_use_and_cost(span.attributes)
                            span.set_attributes(**cost_info.model_dump())

                    except (json.JSONDecodeError, TypeError, AttributeError):
                        logger.warning("Failed to parse span data, %s", span)
                        continue

                    all_spans.append(span)

                json.dump(all_spans, f, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            all_spans = []

        return SpanExportResult.SUCCESS


class RichConsoleSpanExporter(SpanExporter):  # noqa: D101
    def __init__(self, agent_framework: AgentFramework, tracing_config: TracingConfig):  # noqa: D107
        self.processor = TelemetryProcessor.create(agent_framework)
        self.console = Console()
        self.tracing_config = tracing_config

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:  # noqa: D102
        for readable_span in spans:
            style = None
            if not readable_span.attributes:
                msg = "Span must have attributes"
                raise ValueError(msg)
            span = AnyAgentSpan(
                name=readable_span.name,
                kind=readable_span.kind,
                context=readable_span.context,
                parent=readable_span.parent,
                start_time=readable_span.start_time,
                end_time=readable_span.end_time,
                events=readable_span.events,
                attributes=dict(
                    readable_span.attributes
                ),  # turn the mapping into a dict so that it's mutable
                status=readable_span.status,
                links=readable_span.links,
                resource=readable_span.resource,
            )
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

                if (
                    span_kind == "LLM"
                    and self.tracing_config.cost_info
                    and span.attributes
                ):
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
        self.instrumentor = _get_instrumenter_by_framework(
            agent_framework
        )  # Fail fast if framework is not supported
        self.tracing_config = tracing_config
        self.trace_filepath: str | None = None
        self._setup_tracing()

    def __del__(self) -> None:
        """Stop the openinference instrumentation when the tracer is deleted."""
        if self.instrumentor:
            self.instrumentor.uninstrument()

    def _setup_tracing(self) -> None:
        """Set up tracing for the agent."""
        tracer_provider = TracerProvider()

        if self.tracing_config.file:
            if not os.path.exists(self.tracing_config.output_dir):
                os.makedirs(self.tracing_config.output_dir)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.trace_filepath = f"{self.tracing_config.output_dir}/{self.agent_framework.name}-{timestamp}.json"
            json_file_exporter = JsonFileSpanExporter(
                agent_framework=self.agent_framework,
                tracing_config=self.tracing_config,
                file_name=self.trace_filepath,
            )
            span_processor = SimpleSpanProcessor(json_file_exporter)
            tracer_provider.add_span_processor(span_processor)

        if self.tracing_config.console:
            processor = BatchSpanProcessor(
                RichConsoleSpanExporter(self.agent_framework, self.tracing_config),
            )
            tracer_provider.add_span_processor(processor)

        trace.set_tracer_provider(tracer_provider)

        self.instrumentor.instrument(tracer_provider=tracer_provider)

    @property
    def is_enabled(self) -> bool:
        """Whether tracing is enabled."""
        return (
            self.tracing_config.file
            or self.tracing_config.console
            or self.tracing_config.cost_info
        )

    def get_trace(self) -> dict[str, Any] | None:
        """Return the trace data if file tracing is enabled."""
        if self.trace_filepath:
            try:
                with open(self.trace_filepath, encoding="utf-8") as f:
                    content = json.load(f)
                return dict(content)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON trace file.")
        return None
