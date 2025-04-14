import os
import json
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    BatchSpanProcessor,
    SpanExportResult,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from any_agent.config import AgentFramework, TracingConfig
from any_agent.telemetry import TelemetryProcessor


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, file_name: str):
        self.file_name = file_name
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                json.dump([], f)

    def export(self, spans) -> None:
        try:
            with open(self.file_name, "r") as f:
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

    def shutdown(self):
        pass


class RichConsoleSpanExporter(SpanExporter):
    def __init__(self, agent_framework: AgentFramework, tracing_config: TracingConfig):
        self.processor = TelemetryProcessor.create(agent_framework=agent_framework)
        self.console = Console()
        self.tracing_config = tracing_config

    def export(self, spans):
        for span in spans:
            style = None
            span_str = span.to_json()
            span_dict = json.loads(span_str)
            try:
                span_kind, interaction = self.processor.extract_interaction(span_dict)

                style = getattr(self.tracing_config, span_kind.lower(), None)

                if not style:
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
                            )
                        )
                    else:
                        self.console.print(f"{key}: {value}")
            except Exception:
                self.console.print_exception()
            if style:
                self.console.rule(style=style)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _get_tracer_provider(
    agent_framework: AgentFramework, output_dir: str, tracing_config: TracingConfig
) -> tuple[TracerProvider, str | None]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    tracer_provider = TracerProvider()

    file_name = f"{output_dir}/{agent_framework.value}-{timestamp}.json"
    json_file_exporter = JsonFileSpanExporter(file_name=file_name)
    span_processor = SimpleSpanProcessor(json_file_exporter)
    tracer_provider.add_span_processor(span_processor)

    # This is what will log all the span info to stdout: We turn off the agent sdk specific logging so that
    # the user sees a similar logging format for whichever agent they are using under the hood.
    processor = BatchSpanProcessor(
        RichConsoleSpanExporter(agent_framework, tracing_config)
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider, file_name


def setup_tracing(
    agent_framework: AgentFramework,
    output_dir: str = "traces",
    tracing_config: TracingConfig | None = None,
) -> str:
    """Setup tracing for `agent_framework` using `openinference.instrumentation`.

    Args:
        agent_framework (AgentFramework): The type of agent being used.
        output_dir (str): The directory where the traces will be stored.
            Defaults to "traces".
    Returns:
        str: The name of the JSON file where traces will be stored.
    """
    tracing_config = tracing_config or TracingConfig()

    tracer_provider, file_name = _get_tracer_provider(
        agent_framework, output_dir, tracing_config
    )
    if agent_framework == AgentFramework.OPENAI:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    elif agent_framework == AgentFramework.SMOLAGENTS:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
    elif agent_framework == AgentFramework.LANGCHAIN:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    elif agent_framework == AgentFramework.LLAMAINDEX:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    else:
        raise NotImplementedError(f"{agent_framework} tracing is not supported.")
    return file_name
