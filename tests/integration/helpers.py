import asyncio
import time
from typing import Any

import httpx
import requests

from any_agent.config import AgentFramework

DEFAULT_SMALL_MODEL_ID = "gpt-4.1-nano"
DEFAULT_MEDIUM_MODEL_ID = "gpt-4.1-mini"


def get_default_agent_model_args(agent_framework: AgentFramework) -> dict[str, Any]:
    # use a function to avoid passing by reference
    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else {}
    )
    model_args["temperature"] = 0.0
    return model_args


DEFAULT_HTTP_KWARGS = {"timeout": 30.0}


def wait_for_server(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    attempts = 0
    while True:
        try:
            # Try to make a basic GET request to check if server is responding
            requests.get(server_url, timeout=1.0)
            return  # noqa: TRY300
        except (requests.RequestException, ConnectionError):
            # Server not ready yet, continue polling
            pass

        time.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)


async def wait_for_server_async(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    attempts = 0

    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Try to make a basic GET request to check if server is responding
                await client.get(server_url, timeout=1.0)
                return  # noqa: TRY300
            except (httpx.RequestError, httpx.TimeoutException):
                # Server not ready yet, continue polling
                pass

            await asyncio.sleep(poll_interval)
            attempts += 1
            if attempts >= max_attempts:
                msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
                raise ConnectionError(msg)
