import re

import requests
from duckduckgo_search import DDGS
from markdownify import markdownify
from requests.exceptions import RequestException


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


def visit_webpage(url: str) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.

    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()  # type: ignore[no-untyped-call]

        markdown_content = re.sub(r"\n{2,}", "\n", markdown_content)

        return _truncate_content(markdown_content, 10000)
    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"
