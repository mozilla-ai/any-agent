from unittest.mock import MagicMock

from agno.models.message import Message

from any_agent.tracing.instrumentation.agno import (
    _AgnoInstrumentor,
    _set_llm_input,
    _set_llm_output,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input([Message(role="user")], span)

    span.set_attribute.assert_called_with(
        "gen_ai.input.messages", '[{"role": "user", "content": null}]'
    )


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(Message(role="assistant"), span)

    span.set_attributes.assert_called_once_with(
        {"gen_ai.usage.input_tokens": 0, "gen_ai.usage.output_tokens": 0}
    )


def test_uninstrument_before_instrument() -> None:
    _AgnoInstrumentor().uninstrument(MagicMock())
