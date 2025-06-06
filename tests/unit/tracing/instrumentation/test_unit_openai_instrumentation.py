from unittest.mock import MagicMock

from agents.tracing.span_data import GenerationSpanData

from any_agent.tracing.instrumentation.openai import (
    _OpenAIAgentsInstrumentor,
    _set_llm_input,
    _set_llm_output,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input(GenerationSpanData(), span)

    span.set_attribute.assert_not_called()


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(GenerationSpanData(), span)

    span.set_attributes.assert_not_called()


def test_uninstrument_before_instrument() -> None:
    _OpenAIAgentsInstrumentor().uninstrument(MagicMock())
