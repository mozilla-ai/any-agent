"""Constants exported for convenience to access span attributes.

Trying to follow OpenTelemetry's [Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
"""

INPUT_COST = "gen_ai.usage.input_cost"
"""Number of dollars spent for the input of the LLM."""

INPUT_MESSAGES = "gen_ai.input.messages"
"""System prompt and user input."""

INPUT_TOKENS = "gen_ai.usage.input_tokens"
"""Number of tokens spent for the input of the LLM."""

MODEL_ID = "gen_ai.request.model"
"""Underlying (LLM) model used by the agent. For example: `gpt-4.1-mini`."""

OPERATION = "gen_ai.operation.name"
"""Indicates the type of span: `call_llm`, `execute_tool`, `invoke_agent`."""

OUTPUT = "gen_ai.output"
"""Used in both LLM Calls and Tool Executions for holding their respective outputs."""

OUTPUT_COST = "gen_ai.usage.output_cost"
"""Number of dollars spent for the output of the LLM."""

OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
"""Number of tokens spent for the output of the LLM."""

OUTPUT_TYPE = "gen_ai.output.type"
"""One of `json` or `text`. If `json`, the value of `OUTPUT` can be passed to `json.loads`."""

TOOL_ARGS = "gen_ai.tool.args"
"""Arguments passed to the executed tool."""

TOOL_DESCRIPTION = "gen_ai.tool.description"
"""Description of the executed tool."""

TOOL_NAME = "gen_ai.tool.name"
"""Name of the executed tool."""
