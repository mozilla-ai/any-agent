import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, assert_never

from litellm.cost_calculator import cost_per_token
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
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
from any_agent.telemetry import TelemetryProcessor


class Span(Protocol):  # noqa: D101
    def to_json(self) -> str: ...  # noqa: D102


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


def extract_token_use_and_cost(attributes: Mapping[str, Any]) -> TokenUseAndCost:
    """Use litellm and openinference keys to extract token use and cost."""
    span_info: dict[str, int | float] = {}

    for key in ["llm.token_count.prompt", "llm.token_count.completion"]:
        if key in attributes:
            name = key.split(".")[-1]
            span_info[f"token_count_{name}"] = int(attributes[key])

    try:
        cost_prompt, cost_completion = cost_per_token(
            model=attributes.get("llm.model_name", ""),
            prompt_tokens=int(span_info.get("token_count_prompt", 0)),
            completion_tokens=int(span_info.get("token_count_completion", 0)),
        )
        span_info["cost_prompt"] = cost_prompt
        span_info["cost_completion"] = cost_completion
    except Exception as e:
        msg = f"Error computing cost_per_token: {e}"
        logger.warning(msg)
        span_info["cost_prompt"] = 0.0
        span_info["cost_completion"] = 0.0

    return TokenUseAndCost.model_validate(span_info)


class JsonFileSpanExporter(SpanExporter):  # noqa: D101
    def __init__(self, file_name: str):  # noqa: D107
        self.file_name = file_name
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                json.dump([], f)

    def export(self, spans: Sequence[Span]) -> SpanExportResult:  # noqa: D102
        try:
            with open(self.file_name) as f:
                all_spans = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_spans = []

        # Add new spans
        for span in spans:
            try:
                # Try to parse the span data from to_json() if it returns a string
                span_data = json.loads(span.to_json())
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If span.to_json() doesn't return valid JSON string
                span_data = {"error": "Could not serialize span", "span_str": str(span)}

            all_spans.append(span_data)

        # Write all spans back to the file as a proper JSON array
        with open(self.file_name, "w") as f:
            json.dump(all_spans, f, indent=2)

        return SpanExportResult.SUCCESS


class RichConsoleSpanExporter(SpanExporter):  # noqa: D101
    def __init__(self, agent_framework: AgentFramework, tracing_config: TracingConfig):  # noqa: D107
        self.processor = TelemetryProcessor.create(agent_framework)
        self.console = Console()
        self.tracing_config = tracing_config

    def export(self, spans: Sequence[Span]) -> SpanExportResult:  # noqa: D102
        for span in spans:
            style = None
            span_str = span.to_json()
            span_dict = json.loads(span_str)
            try:
                span_kind, interaction = self.processor.extract_interaction(span_dict)

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
                    cost_info = extract_token_use_and_cost(
                        span_dict.get("attributes", {})
                    )
                    for key, value in cost_info.model_dump().items():
                        self.console.print(f"{key}: {value}")

            except Exception:
                self.console.print_exception()
            if style:
                self.console.rule(style=style)
        return SpanExportResult.SUCCESS


class CostTrackingExporter(SpanExporter):
    """A span exporter that only tracks the total cost without displaying anything."""

    def __init__(self, agent_framework: AgentFramework):  # noqa: D107
        self.processor = TelemetryProcessor.create(agent_framework)
        self.costs: list[TokenUseAndCost] = []

    def export(self, spans: Sequence[Span]) -> SpanExportResult:  # noqa: D102
        for span in spans:
            span_str = span.to_json()
            span_dict = json.loads(span_str)
            span_kind, _ = self.processor.extract_interaction(span_dict)
            if span_kind == "LLM":
                cost_info = extract_token_use_and_cost(span_dict.get("attributes", {}))
                self.costs.append(cost_info)
        return SpanExportResult.SUCCESS

    def get_cost_summary(self) -> TotalTokenUseAndCost:
        """Return the current total cost and token usage statistics."""
        total_cost = sum(cost.cost_prompt + cost.cost_completion for cost in self.costs)
        total_tokens = sum(
            cost.token_count_prompt + cost.token_count_completion for cost in self.costs
        )
        total_token_count_prompt = sum(cost.token_count_prompt for cost in self.costs)
        total_token_count_completion = sum(
            cost.token_count_completion for cost in self.costs
        )
        total_cost_prompt = sum(cost.cost_prompt for cost in self.costs)
        total_cost_completion = sum(cost.cost_completion for cost in self.costs)
        return TotalTokenUseAndCost(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_token_count_prompt=total_token_count_prompt,
            total_token_count_completion=total_token_count_completion,
            total_cost_prompt=total_cost_prompt,
            total_cost_completion=total_cost_completion,
        )


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
            json_file_exporter = JsonFileSpanExporter(file_name=self.trace_filepath)
            span_processor = SimpleSpanProcessor(json_file_exporter)
            tracer_provider.add_span_processor(span_processor)

        if self.tracing_config.console:
            processor = BatchSpanProcessor(
                RichConsoleSpanExporter(self.agent_framework, self.tracing_config),
            )
            tracer_provider.add_span_processor(processor)

        if self.tracing_config.cost_info:
            self.cost_tracker = CostTrackingExporter(self.agent_framework)
            cost_processor = SimpleSpanProcessor(self.cost_tracker)
            tracer_provider.add_span_processor(cost_processor)

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
                with open(self.trace_filepath) as f:
                    content = json.load(f)
                return dict(content)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON trace file.")
        return None

    def get_cost_summary(self) -> TotalTokenUseAndCost:
        """Return the current total cost and token usage statistics."""
        return self.cost_tracker.get_cost_summary()
