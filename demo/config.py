import os
import tempfile
from datetime import datetime, timedelta
from typing import Annotated

import geocoder
from pydantic import AfterValidator, BaseModel, ConfigDict, FutureDatetime, PositiveInt
from rich.prompt import Prompt

from any_agent import AgentFramework
from any_agent.config import AgentConfig
from any_agent.logging import logger

INPUT_PROMPT_TEMPLATE = """
According to the forecast, what will be the best spot to surf around {LOCATION},
in a {MAX_DRIVING_HOURS} hour driving radius,
at {DATE}?"
""".strip()


def validate_prompt(value) -> str:
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}", "{DATE}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    input_prompt_template: Annotated[str, AfterValidator(validate_prompt)] = (
        INPUT_PROMPT_TEMPLATE
    )

    framework: AgentFramework

    main_agent: AgentConfig

    evaluation_model: str | None = None
    evaluation_criteria: list[dict[str, str]] | None = None
