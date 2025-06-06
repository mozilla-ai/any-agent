from unittest.mock import MagicMock

from llama_index.core.agent.workflow.workflow_events import AgentOutput
from llama_index.core.base.llms.types import ChatMessage

from any_agent.tracing.instrumentation.llama_index import (
    _LlamaIndexInstrumentor,
    _set_llm_input,
    _set_llm_output,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input([ChatMessage()], span)

    span.set_attribute.assert_called_with(
        "gen_ai.input.messages", '[{"role": "user", "content": "No content"}]'
    )


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(
        AgentOutput(
            response=ChatMessage(), tool_calls=[], raw=None, current_agent_name="foo"
        ),
        span,
    )

    span.set_attributes.assert_not_called()


def test_uninstrument_before_instrument() -> None:
    _LlamaIndexInstrumentor().uninstrument(MagicMock())
