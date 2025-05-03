from collections.abc import Mapping

from litellm.cost_calculator import cost_per_token
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, ConfigDict

from any_agent.logging import logger


class TokenUseAndCost(BaseModel):
    """Token use and cost information."""

    token_count_prompt: int
    token_count_completion: int
    cost_prompt: float
    cost_completion: float

    model_config = ConfigDict(extra="forbid")


class TotalTokenUseAndCost(BaseModel):
    """Total token use and cost information."""

    total_token_count_prompt: int
    total_token_count_completion: int
    total_cost_prompt: float
    total_cost_completion: float

    total_cost: float
    total_tokens: int

    model_config = ConfigDict(extra="forbid")


def extract_token_use_and_cost(
    attributes: Mapping[str, AttributeValue],
) -> TokenUseAndCost:
    """Use litellm and openinference keys to extract token use and cost."""
    span_info: dict[str, AttributeValue] = {}

    for key in ["llm.token_count.prompt", "llm.token_count.completion"]:
        if key in attributes:
            name = key.split(".")[-1]
            span_info[f"token_count_{name}"] = attributes[key]
    try:
        cost_prompt, cost_completion = cost_per_token(
            model=str(attributes.get("llm.model_name", "")),
            prompt_tokens=int(attributes.get("llm.token_count.prompt", 0)),  # type: ignore[arg-type]
            completion_tokens=int(span_info.get("llm.token_count.completion", 0)),  # type: ignore[arg-type]
        )
        span_info["cost_prompt"] = cost_prompt
        span_info["cost_completion"] = cost_completion
    except Exception as e:
        msg = f"Error computing cost_per_token: {e}"
        logger.warning(msg)
        span_info["cost_prompt"] = 0.0
        span_info["cost_completion"] = 0.0

    return TokenUseAndCost.model_validate(span_info)
