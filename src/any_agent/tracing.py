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
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.config import AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.telemetry import TelemetryProcessor


class Span(Protocol):  # noqa: D101
    def to_json(self) -> str: ...  # noqa: D102


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

    def _extract_token_use_and_cost(
        self, attributes: Mapping[str, Any]
    ) -> dict[str, int | float]:
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
            span_info["cost_prompt ($)"] = cost_prompt
            span_info["cost_completion ($)"] = cost_completion
        except Exception as e:
            msg = f"Error computing cost_per_token: {e}"
            logger.warning(msg)

        return span_info

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
                    cost_info = self._extract_token_use_and_cost(
                        span_dict.get("attributes", {})
                    )
                    for key, value in cost_info.items():
                        self.console.print(f"{key}: {value}")

            except Exception:
                self.console.print_exception()
            if style:
                self.console.rule(style=style)
        return SpanExportResult.SUCCESS


def _get_tracer_provider(
    agent_framework: AgentFramework,
    tracing_config: TracingConfig,
) -> tuple[TracerProvider, str]:
    tracer_provider = TracerProvider()
    if tracing_config.output_dir is not None:
        if not os.path.exists(tracing_config.output_dir):
            os.makedirs(tracing_config.output_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = (
            f"{tracing_config.output_dir}/{agent_framework.name}-{timestamp}.json"
        )
        json_file_exporter = JsonFileSpanExporter(file_name=file_name)
        span_processor = SimpleSpanProcessor(json_file_exporter)
        tracer_provider.add_span_processor(span_processor)

    # This is what will log all the span info to stdout: We turn off the agent sdk specific logging so that
    # the user sees a similar logging format for whichever agent they are using under the hood.
    processor = BatchSpanProcessor(
        RichConsoleSpanExporter(agent_framework, tracing_config),
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider, file_name


def setup_tracing(
    agent_framework: AgentFramework,
    tracing_config: TracingConfig,
) -> str:
    """Set up tracing for `agent_framework` using `openinference.instrumentation`.

    Args:
        agent_framework (AgentFramework): The type of agent being used.
        tracing_config (TracingConfig): Configuration for tracing, including output directory and styles.

    Returns:
        str: The name of the JSON file where traces will be stored.

    """
    agent_framework_ = AgentFramework.from_string(agent_framework)

    tracer_provider, file_name = _get_tracer_provider(
        agent_framework,
        tracing_config,
    )

    instrumenter = _get_instrumenter_by_framework(agent_framework_)
    instrumenter.instrument(tracer_provider=tracer_provider)

    return file_name


class Instrumenter(Protocol):  # noqa: D101
    def instrument(self, *, tracer_provider: TracerProvider) -> None: ...  # noqa: D102


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

    if framework is AgentFramework.GOOGLE or framework is AgentFramework.AGNO:
        msg = f"{framework} tracing is not supported."
        raise NotImplementedError(msg)

    assert_never(framework)
