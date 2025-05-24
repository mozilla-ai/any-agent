from unittest.mock import MagicMock

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from any_agent.tracing.instrumentation.google import (
    _GoogleADKInstrumentor,
    _set_llm_input,
    _set_llm_output,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input(LlmRequest(), span)

    span.set_attribute.assert_not_called()


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(LlmResponse(), span)

    span.set_attributes.assert_not_called()


def test_uninstrument_before_instrument() -> None:
    _GoogleADKInstrumentor().uninstrument()
