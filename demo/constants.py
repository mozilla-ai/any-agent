import os

from tools import (
    get_area_lat_lon,
    get_wave_forecast,
    get_wind_forecast,
)

from any_agent.evaluation import EvaluationCase
from any_agent.logging import logger
from any_agent.tools.web_browsing import search_tavily, search_web, visit_webpage

MODEL_OPTIONS = [
    # "huggingface/novita/deepseek-ai/DeepSeek-V3",
    # "huggingface/novita/meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "gemini/gemini-2.0-flash-lite",
    "gemini/gemini-2.0-flash",
    # "huggingface/Qwen/Qwen3-32B", # right now throwing an internal error, but novita qwen isn't supporting tool calling
]

# Novita was the only HF based provider that worked.

# Hugginface API Provider Error:
# Must alternate between assistant/user, which meant that the 'tool' role made it puke


DEFAULT_EVALUATION_CASE = EvaluationCase(
    llm_judge=MODEL_OPTIONS[0],
    checkpoints=[
        {
            "criteria": "Check if the agent considered at least three surf spot options",
            "points": 1,
        },
        {
            "criteria": "Check if the agent gathered wind forecasts for each surf spot being evaluated.",
            "points": 1,
        },
        {
            "criteria": "Check if the agent gathered wave forecasts for each surf spot being evaluated.",
            "points": 1,
        },
        {
            "criteria": "Check if the agent used any web search tools to explore which surf spots should be considered",
            "points": 1,
        },
        {
            "criteria": "Check if the final answer contains any description about the weather (air temp, chance of rain, etc) at the chosen location",
            "points": 1,
        },
        {
            "criteria": "Check if the final answer includes one of the surf spots evaluated by tools",
            "points": 1,
        },
        {
            "criteria": "Check if the final answer includes information about some alternative surf spots if the user is not satisfied with the chosen one",
            "points": 1,
        },
    ],
)


DEFAULT_TOOLS = [
    get_wind_forecast,
    get_wave_forecast,
    get_area_lat_lon,
    search_web,
    visit_webpage,
]
if os.getenv("TAVILY_API_KEY"):
    DEFAULT_TOOLS.append(search_tavily)
else:
    logger.warning("TAVILY_API_KEY not set, skipping Tavily search tool")
