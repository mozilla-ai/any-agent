import os
import json
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from any_agent import AgentFramework


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, file_name: str):
        self.file_name = file_name
        # Initialize with an empty array if file doesn't exist
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                json.dump([], f)

    def export(self, spans) -> None:
        # Read existing spans
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


def get_tracer_provider(
    project_name: str, agent_framework: AgentFramework, output_dir: str = "output"
) -> tuple[TracerProvider, str | None]:
    """
    Create a tracer_provider based on the selected mode.

    Args:
        project_name: Name of the project for tracing
        output_dir: The directory where the telemetry output will be stored. Defaults to "output".

    Returns:
        tracer_provider: The configured tracer provider
        file_name: The name of the JSON file where telemetry will be stored
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

    file_name = f"{output_dir}/{agent_framework.value}-{project_name}-{timestamp}.json"
    json_file_exporter = JsonFileSpanExporter(file_name=file_name)
    span_processor = SimpleSpanProcessor(json_file_exporter)
    tracer_provider.add_span_processor(span_processor)

    return tracer_provider, file_name


def setup_tracing(
    tracer_provider: TracerProvider, agent_framework: AgentFramework
) -> None:
    """Setup tracing for `agent_framework` by instrumenting `trace_provider`.

    Args:
        tracer_provider (TracerProvider): The configured tracer provider from
            [get_tracer_provider][any_agent.tracing.get_tracer_provider].
        agent_framework (AgentFramework): The type of agent being used.
    """
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
