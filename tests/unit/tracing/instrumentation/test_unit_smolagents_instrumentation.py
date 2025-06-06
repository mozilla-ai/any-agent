from unittest.mock import MagicMock

from smolagents.models import ChatMessage

from any_agent.tracing.instrumentation.smolagents import (
    _set_llm_input,
    _set_llm_output,
    _SmolagentsInstrumentor,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input([], span)

    span.set_attribute.assert_not_called()


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(ChatMessage("assistant"), span)

    span.set_attributes.assert_not_called()


def test_uninstrument_before_instrument() -> None:
    _SmolagentsInstrumentor().uninstrument(MagicMock())
