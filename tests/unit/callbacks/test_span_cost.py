from unittest.mock import MagicMock

from any_agent.callbacks.span_cost import AddCostInfo
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID
from any_agent.tracing.attributes import GenAI


def test_span_cost() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        GenAI.REQUEST_MODEL: DEFAULT_SMALL_MODEL_ID,
        GenAI.USAGE_INPUT_TOKENS: 100,
        GenAI.USAGE_OUTPUT_TOKENS: 1000,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_COST] > 0
    assert call_args[GenAI.USAGE_OUTPUT_COST] > 0


def test_span_cost_missing_input() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        GenAI.REQUEST_MODEL: DEFAULT_SMALL_MODEL_ID,
        GenAI.USAGE_OUTPUT_TOKENS: 1000,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_COST] == 0
    assert call_args[GenAI.USAGE_OUTPUT_COST] > 0


def test_span_cost_missing_output() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        GenAI.REQUEST_MODEL: DEFAULT_SMALL_MODEL_ID,
        GenAI.USAGE_INPUT_TOKENS: 100,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args[GenAI.USAGE_INPUT_COST] > 0
    assert call_args[GenAI.USAGE_OUTPUT_COST] == 0


def test_span_cost_missing_all() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        GenAI.REQUEST_MODEL: DEFAULT_SMALL_MODEL_ID,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    context.current_span.set_attributes.assert_not_called()
