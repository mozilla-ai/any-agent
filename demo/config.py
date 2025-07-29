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
from any_agent.evaluation import EvaluationCase
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
    # the default should be the current date + 1 day
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


def get_litellm_model_id(agent_name) -> str:
    """Ask the user to input a model_id string. Validate it using the litellm.validate_environment function"""
    from litellm.utils import validate_environment

    prompt = f"Enter a valid model_id for agent {agent_name} using LiteLLM syntax"
    default_val = "openai/gpt-4o"
    model_id = Prompt.ask(prompt, default=default_val)
    # make a call to validate the model id: this will throw an error if the model id is not valid
    get_llm_provider(model=model_id)
    # make a call to validate that the environment is correct for the model id
    env_check = validate_environment(model_id)
    if not env_check["keys_in_environment"]:
        msg = f"{env_check['missing_keys']} needed for {model_id}"
        raise ValueError(msg)
    return model_id


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
    managed_agents: list[AgentConfig] | None = None

    evaluation_cases: list[EvaluationCase] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a Config instance from a dictionary.

        Args:
            data (dict): A dictionary containing the configuration data.

        Returns:
            Config: A new Config instance populated with values from the dictionary.

        """
        # for each tool listed in main_agent.tools, use import lib to import it and replace the str with the callable
        callables = []
        if data.get("main_agent") is None:
            data["main_agent"] = {}
        if not data["main_agent"].get("model_id"):
            data["main_agent"]["model_id"] = get_litellm_model_id("main_agent")
        else:
            logger.info(f"Main agent using model_id {data['main_agent']['model_id']}")
        for tool in data["main_agent"].get("tools", []):
            if isinstance(tool, str):
                module_name, func_name = tool.rsplit(".", 1)
                module = __import__(module_name, fromlist=[func_name])
                callables.append(getattr(module, func_name))
            else:
                # this means it must be an MCPStdioParams
                # For the purposes of this demo, currently we just look for the filesystem MCP which we have a placeholder
                # for the path variable (which controls which dirs the MCP will have access to).
                mcp_tool = set_mcp_settings(tool)
                callables.append(mcp_tool)
        data["main_agent"]["tools"] = callables
        for agent in data.get("managed_agents", []):
            if agent.get("model_id") is None:
                agent["model_id"] = get_litellm_model_id(
                    agent.get("name", "managed_agent")
                )
            else:
                logger.info(f"Agent {agent['name']} using model_id {agent['model_id']}")
            callables = []
            for tool in agent.get("tools", []):
                if isinstance(tool, str):
                    module_name, func_name = tool.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[func_name])
                    callables.append(getattr(module, func_name))
                else:
                    # this means it must be an MCPStdioParams
                    mcp_tool = set_mcp_settings(tool)
                    callables.append(mcp_tool)
            agent["tools"] = callables
        if not data.get("framework"):
            data["framework"] = ask_framework()
        else:
            logger.info(f"Using framework {data['framework']}")
        if not data.get("location"):
            data["location"] = location_picker()
        else:
            logger.info(f"Using location {data['location']}")
        if not data.get("max_driving_hours"):
            data["max_driving_hours"] = max_driving_hours_picker()
        else:
            logger.info(f"Using max driving hours {data['max_driving_hours']}")
        if not data.get("date"):
            data["date"] = date_picker()
        else:
            logger.info(f"Using date {data['date']}")

        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """With open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)    yaml_path: Path to the YAML configuration file

        Returns:
            Config: A new Config instance populated with values from the YAML file

        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
