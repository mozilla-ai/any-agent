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
    _GoogleADKInstrumentor().uninstrument(MagicMock())


def test_instrument_uninstrument() -> None:
    """Regression test for https://github.com/mozilla-ai/any-agent/issues/467"""
    agent = MagicMock()
    agent._agent.before_model_callback = None
    agent._agent.after_model_callback = None
    agent._agent.before_tool_callback = None
    agent._agent.after_tool_callback = None
    instrumentor = _GoogleADKInstrumentor()

    instrumentor.instrument(agent)
    assert callable(agent._agent.before_model_callback)
    assert callable(agent._agent.after_model_callback)
    assert callable(agent._agent.before_tool_callback)
    assert callable(agent._agent.after_tool_callback)

    instrumentor.uninstrument(agent)
    assert agent._agent.before_model_callback is None
    assert agent._agent.after_model_callback is None
    assert agent._agent.before_tool_callback is None
    assert agent._agent.after_tool_callback is None
