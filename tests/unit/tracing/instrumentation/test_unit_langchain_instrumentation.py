from unittest.mock import MagicMock

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.outputs.generation import Generation

from any_agent.tracing.instrumentation.langchain import (
    _LangChainInstrumentor,
    _set_llm_input,
    _set_llm_output,
)


def test_set_llm_input_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_input([[BaseMessage(content="foo", type="human")]], span)

    span.set_attribute.assert_called_with(
        "gen_ai.input.messages", '[{"role": "user", "content": "foo"}]'
    )


def test_set_llm_output_missing_fields() -> None:
    """It should not fail when missing fields."""
    span = MagicMock()
    _set_llm_output(LLMResult(generations=[[Generation(text="")]]), span)

    span.set_attributes.assert_not_called()


def test_uninstrument_before_instrument() -> None:
    _LangChainInstrumentor().uninstrument(MagicMock())
