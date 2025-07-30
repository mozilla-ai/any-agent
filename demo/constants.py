import os

from tools import (
    get_area_lat_lon,
    get_wave_forecast,
    get_wind_forecast,
)

from any_agent.logging import logger
from any_agent.tools.web_browsing import search_tavily, search_web, visit_webpage

INPUT_PROMPT_TEMPLATE = """
According to the forecast, what will be the best spot to surf around {LOCATION},
in a {MAX_DRIVING_HOURS} hour driving radius,
at {DATE}?"
""".strip()

MODEL_OPTIONS = [
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
]

DEFAULT_EVALUATION_CASES = [
    "Check if the agent considered at least three surf spot options",
    "Check if the agent used any web search tools to explore which surf spots should be considered",
    "Check if the agent gathered wind forecasts for each surf spot being evaluated.",
    "Check if the agent gathered wave forecasts for each surf spot being evaluated.",
    "Check if the final answer contains any description about the weather (air temp, chance of rain, etc) at the chosen location",
]

DEFAULT_TOOLS = [
    get_wind_forecast,
    get_wave_forecast,
    get_area_lat_lon,
    visit_webpage,
]
if os.getenv("TAVILY_API_KEY"):
    DEFAULT_TOOLS.append(search_tavily)
else:
    DEFAULT_TOOLS.append(search_web)
    logger.warning("TAVILY_API_KEY not set, skipping Tavily search tool")
