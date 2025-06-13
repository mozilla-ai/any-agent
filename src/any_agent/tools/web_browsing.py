import os
import re

import requests
from duckduckgo_search import DDGS
from markdownify import markdownify
from requests.exceptions import RequestException

try:
    from tavily.tavily import TavilyClient

    tavily_available = True
except ImportError:
    tavily_available = False


def _truncate_content(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content
    return (
        content[: max_length // 2]
        + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        + content[-max_length // 2 :]
    )


def search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query (think a Google search) then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    ddgs = DDGS()
    results = ddgs.text(query, max_results=10)
    return "\n".join(
        f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
    )


def visit_webpage(url: str, timeout: int = 30) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.
        timeout: The timeout in seconds for the request.

    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()  # type: ignore[no-untyped-call]

        markdown_content = re.sub(r"\n{2,}", "\n", markdown_content)

        return _truncate_content(markdown_content, 10000)
    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"


def search_tavily(query: str, include_images: bool = False) -> str:
    """Perform a Tavily web search based on your query and return the top search results.

    See https://blog.tavily.com/getting-started-with-the-tavily-search-api for more information.

    Args:
        query (str): The search query to perform.
        include_images (bool): Whether to include images in the results.

    Returns:
        The top search results as a formatted string.

    """
    if not tavily_available:
        msg = "You need to `pip install 'tavily-python'` to use this tool"
        raise ImportError(msg)
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY environment variable not set."
    try:
        client = TavilyClient(api_key)
        response = client.search(query, include_images=include_images)
        results = response.get("results", [])
        output = []
        for result in results:
            output.append(
                f"[{result.get('title', 'No Title')}]({result.get('url', '#')})\n{result.get('content', '')}"
            )
        if include_images and "images" in response:
            output.append("\nImages:")
            for image in response["images"]:
                output.append(image)
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error performing Tavily search: {e!s}"
