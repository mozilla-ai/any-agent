import os
import json
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from any_agent import AgentFramework
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


def create_console_span_formatter(agent_framework: AgentFramework) -> callable:
    """
    This function creates the correct span formatter for each given framework,
    Since even though we are using openinference there are still differences
    in how to extract the correct info.

    Args:
        agent_framework (AgentFramework): The type of agent being used.
    Returns:
        callable: A function that formats spans for console output.
    """

    def console_span_formatter(span: ReadableSpan) -> str:
        """
        Format a span for console output. The input and output signature is set by opentelemetry
        Args:
            span (ReadableSpan): The span to format.
        Returns:
            str: The formatted span string.
        """
        span_str = span.to_json()
        span_dict = json.loads(span_str)

        processor = TelemetryProcessor.create(agent_framework=agent_framework)
        try:
            interaction = processor.extract_interaction(span_dict)
            output_str = "-" * 50 + "\n"
            for key, value in interaction.items():
                output_str += f"{key}: {value}\n"
        except Exception as e:
            output_str = f"Error processing span: {e}\n"
            output_str += f"Span data: {span_str}\n"
        output_str += "\n"
        output_str += "-" * 50 + "\n"
        return output_str

    return console_span_formatter


def _get_tracer_provider(
    agent_framework: AgentFramework, output_dir: str = "output"
) -> tuple[TracerProvider, str | None]:
    """
    Create a tracer_provider that will write to `output_dir`.

    Args:
        output_dir: The directory where the traces will be stored.
            Defaults to "output".

    Returns:
        tracer_provider: The configured tracer provider
        file_name: The name of the JSON file where traces will be stored
    """
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
        ConsoleSpanExporter(formatter=create_console_span_formatter(agent_framework))
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider, file_name


def setup_tracing(agent_framework: AgentFramework, output_dir: str = "traces") -> str:
    """Setup tracing for `agent_framework` using `openinference.instrumentation`.

    Args:
        agent_framework (AgentFramework): The type of agent being used.
        output_dir (str): The directory where the traces will be stored.
            Defaults to "traces".
    Returns:
        str: The name of the JSON file where traces will be stored.
    """
    tracer_provider, file_name = _get_tracer_provider(agent_framework, output_dir)
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
