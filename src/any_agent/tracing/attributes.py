from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)


class GenAI:
    """Constants exported for convenience to access span attributes.

    Trying to follow OpenTelemetry's [Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

    We import the constants from `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`
    whenever is possible.

    We only expose the keys that we currently use in `any-agent`.
    """

    AGENT_DESCRIPTION = GEN_AI_AGENT_DESCRIPTION
    """Free-form description of the GenAI agent provided by the application."""

    AGENT_NAME = GEN_AI_AGENT_NAME
    """Human-readable name of the GenAI agent provided by the application."""

    INPUT_MESSAGES = "gen_ai.input.messages"
    """System prompt and user input."""

    OPERATION_NAME = GEN_AI_OPERATION_NAME
    """The name of the operation being performed."""

    OUTPUT = "gen_ai.output"
    """Used in both LLM Calls and Tool Executions for holding their respective outputs."""

    OUTPUT_TYPE = GEN_AI_OUTPUT_TYPE
    """Represents the content type requested by the client."""

    REQUEST_MODEL = GEN_AI_REQUEST_MODEL
    """The name of the GenAI model a request is being made to."""

    TOOL_ARGS = "gen_ai.tool.args"
    """Arguments passed to the executed tool."""

    TOOL_DESCRIPTION = GEN_AI_TOOL_DESCRIPTION
    """The tool description."""

    TOOL_NAME = GEN_AI_TOOL_NAME
    """Name of the tool utilized by the agent."""

    USAGE_INPUT_COST = "gen_ai.usage.input_cost"
    """Dollars spent for the input of the LLM."""

    USAGE_INPUT_TOKENS = GEN_AI_USAGE_INPUT_TOKENS
    """The number of tokens used in the GenAI input (prompt)."""

    USAGE_OUTPUT_COST = "gen_ai.usage.output_cost"
    """Dollars spent for the output of the LLM."""

    USAGE_OUTPUT_TOKENS = GEN_AI_USAGE_OUTPUT_TOKENS
    """The number of tokens used in the GenAI response (completion)."""
