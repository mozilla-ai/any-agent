import os
import tempfile
from datetime import datetime, timedelta
from typing import Annotated

import geocoder
import yaml
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
)
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


def ask_framework() -> AgentFramework:
    """Ask the user which framework they would like to use. They must select one of the Agent Frameworks"""
    frameworks = [framework.name for framework in AgentFramework]
    frameworks_str = "\n".join(
        [f"{i}: {framework}" for i, framework in enumerate(frameworks)]
    )
    prompt = f"Select the agent framework to use:\n{frameworks_str}\n"
    choice = Prompt.ask(prompt, default="0")
    try:
        choice = int(choice)
        if choice < 0 or choice >= len(frameworks):
            raise ValueError("Invalid choice")
        return AgentFramework[frameworks[choice]]
    except ValueError:
        raise ValueError("Invalid choice")


def date_picker() -> FutureDatetime:
    """Ask the user to select a date in the future. The date must be at least 1 day in the future."""
    prompt = "Select a date in the future (YYYY-MM-DD-HH)"
    now = datetime.now()
    default_val = (now + timedelta(days=1)).strftime("%Y-%m-%d-%H")
    date_str = Prompt.ask(prompt, default=default_val)
    try:
        year, month, day, hour = map(int, date_str.split("-"))
        date = datetime(year, month, day, hour)
        return date
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD-HH.")


def location_picker() -> str:
    """Ask the user to input a location. By default use the current location based on the IP address."""
    prompt = "Enter a location"
    g = geocoder.ip("me")
    default_val = f"{g.city} {g.state}, {g.country}"
    location = Prompt.ask(prompt, default=default_val)
    if not location:
        raise ValueError("location cannot be empty")
    return location


def max_driving_hours_picker() -> int:
    """Ask the user to input the maximum driving hours. The default is 2 hours."""
    prompt = "Enter the maximum driving hours"
    default_val = str(2)
    max_driving_hours = Prompt.ask(prompt, default=default_val)
    try:
        max_driving_hours = int(max_driving_hours)
        if max_driving_hours <= 0:
            raise ValueError("Invalid choice")
        return max_driving_hours
    except ValueError:
        raise ValueError("Invalid choice")


def set_mcp_settings(tool):
    logger.info(
        f"This MCP uses {tool['command']}. If you don't have this set up this will not work"
    )
    if "mcp/filesystem" not in tool["args"]:
        msg = "The only MCP that this demo supports is the filesystem MCP"
        raise ValueError(msg)
    if not any("{{ path_variable }}" in arg for arg in tool["args"]):
        msg = "The filesystem MCP must have { path_variable } in the args list"
        raise ValueError(msg)
    for idx, item in enumerate(tool["args"]):
        if "{{ path_variable }}" in item:
            default_val = os.path.join(tempfile.gettempdir(), "surf_spot_finder")
            answer = Prompt.ask(
                "Please enter the path you'd like the Filesystem MCP to access",
                default=default_val,
            )
            os.makedirs(answer, exist_ok=True)
            tool["args"][idx] = item.replace("{{ path_variable }}", answer)
    return tool


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
