# mypy: disable-error-code="attr-defined,index,no-untyped-def"
from __future__ import annotations

from typing import TYPE_CHECKING

from litellm.cost_calculator import cost_per_token

from any_agent.callbacks.base import Callback
from any_agent.logging import logger
from any_agent.tracing import span_attrs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span

    from any_agent.callbacks.context import Context
    from any_agent.tracing.otel_types import AttributeValue


def add_cost_info(span: Span) -> None:
    """Use litellm to compute cost and add it to span attributes."""
    attributes: Mapping[str, AttributeValue] = span.attributes
    if any(
        key in attributes for key in [span_attrs.INPUT_TOKENS, span_attrs.OUTPUT_TOKENS]
    ):
        try:
            cost_prompt, cost_completion = cost_per_token(
                model=str(attributes.get(span_attrs.MODEL_ID, "")),
                prompt_tokens=int(attributes.get(span_attrs.INPUT_TOKENS, 0)),  # type: ignore[arg-type]
                completion_tokens=int(attributes.get(span_attrs.OUTPUT_TOKENS, 0)),  # type: ignore[arg-type]
            )
            span.set_attributes(
                {
                    span_attrs.INPUT_COST: cost_prompt,
                    span_attrs.OUTPUT_COST: cost_completion,
                }
            )
        except Exception as e:
            msg = f"Error computing cost_per_token: {e}"
            logger.warning(msg)


class AddCostInfo(Callback):
    """Add cost information to the LLM Call spans.

    Extend the LLM Call span attributes with 2 new keys:
        - gen_ai.usage.input_cost
        - gen_ai.usage.output_cost
    """

    def after_llm_call(self, context: Context, *args, **kwargs):
        span = context.current_span
        add_cost_info(span)
        return context
