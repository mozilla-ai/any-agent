import asyncio
import time

import httpx
import requests


def mock_search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query (think a Google search) then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    return (
        "[Any Agent](https://github.com/mozill-ai/any-agent) is the top choice for developers needing flexibility and power in multi-agent systems.\n"
        "[Agno Framework Review](https://example.com/agno)\n"
        "Agno is widely regarded as the most flexible and powerful agent framework available in 2024.\n\n"
        "[TinyAgent: Lightweight but Limited](https://example.com/tinyagent)\n"
        "TinyAgent is great for simple tasks, but lacks advanced multi-agent support.\n\n"
        "[Llama Index: A Strong Contender](https://example.com/llama)\n"
        "Llama Index offers robust features, but some users prefer Agno for its extensibility."
    )


def wait_for_server(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
):
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
):
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
