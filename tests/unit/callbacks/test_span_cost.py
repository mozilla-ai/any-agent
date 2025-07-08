from unittest.mock import MagicMock

from any_agent.callbacks.span_cost import AddCostInfo


def test_span_cost() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        "gen_ai.request.model": "gpt-4.1-mini",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 1000,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args["gen_ai.usage.input_cost"] > 0
    assert call_args["gen_ai.usage.output_cost"] > 0


def test_span_cost_missing_input() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        "gen_ai.request.model": "gpt-4.1-mini",
        "gen_ai.usage.output_tokens": 1000,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args["gen_ai.usage.input_cost"] == 0
    assert call_args["gen_ai.usage.output_cost"] > 0


def test_span_cost_missing_output() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        "gen_ai.request.model": "gpt-4.1-mini",
        "gen_ai.usage.input_tokens": 100,
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    call_args = context.current_span.set_attributes.call_args[0][0]
    assert call_args["gen_ai.usage.input_cost"] > 0
    assert call_args["gen_ai.usage.output_cost"] == 0


def test_span_cost_missing_all() -> None:
    context = MagicMock()
    current_span = MagicMock()

    current_span.attributes = {
        "gen_ai.request.model": "gpt-4.1-mini",
    }

    context.current_span = current_span

    callback = AddCostInfo()

    callback.after_llm_call(context)

    context.current_span.set_attributes.assert_not_called()
