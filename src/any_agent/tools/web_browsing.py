import json
import os
import re
from typing import Any

import requests
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
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "You need to `pip install 'duckduckgo_search'` to use this tool"
        raise ImportError(msg) from e

    ddgs = DDGS()
    results = ddgs.text(query, max_results=10)
    return "\n".join(
        f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
    )


def visit_webpage(url: str, timeout: int = 30, max_length: int = 10000) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.
        timeout: The timeout in seconds for the request.
        max_length: The maximum number of characters of text that can be returned (default=10000).
                    If max_length==-1, text is not truncated and the full webpage is returned.

    """
    try:
        from markdownify import markdownify  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "You need to `pip install 'markdownify'` to use this tool"
        raise ImportError(msg) from e

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()

        markdown_content = re.sub(r"\n{2,}", "\n", markdown_content)

        if max_length == -1:
            return str(markdown_content)
        return _truncate_content(markdown_content, max_length)
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
    try:
        from tavily.tavily import TavilyClient
    except ImportError as e:
        msg = "You need to `pip install 'tavily-python'` to use this tool"
        raise ImportError(msg) from e

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


def search_youcom(query: str, max_results: int = 10, timeout: int = 30) -> str:
    """Perform a You.com web search based on your query and return the top search results.

    Uses the keyless free profile of the You.com MCP server, so no API key is
    required. Set YDC_API_KEY to use the authenticated endpoint instead.

    Args:
        query (str): The search query to perform.
        max_results (int): The maximum number of results to return (default=10).
        timeout (int): The timeout in seconds for each HTTP request (default=30).

    Returns:
        The top search results as a formatted string.

    """
    api_key = os.getenv("YDC_API_KEY")
    url = (
        "https://api.you.com/mcp" if api_key else "https://api.you.com/mcp?profile=free"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _request(payload: dict[str, Any]) -> Any:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        if "Mcp-Session-Id" not in headers and response.headers.get("Mcp-Session-Id"):
            headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        if response.status_code in (202, 204) or not response.text.strip():
            # Accepted notifications carry no payload.
            return None
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for line in reversed(response.text.splitlines()):
                if line.startswith("data:"):
                    data = line.removeprefix("data:").strip()
                    if data:
                        return json.loads(data)
            return None
        return response.json()

    try:
        _request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "any-agent", "version": "1.0"},
                },
            }
        )
        _request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        response = _request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "you-search",
                    "arguments": {"query": query, "count": max_results},
                },
            }
        )
        call_result = (response or {}).get("result", {})
        blocks = [block.get("text", "") for block in call_result.get("content", [])]
        text = "\n".join(block for block in blocks if block)
        try:
            results = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            results = None
        if isinstance(results, dict):
            # The you-search tool returns {"results": {"web": [...]}}.
            results = (results.get("results") or {}).get("web")
        if isinstance(results, list):
            output = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                snippet = result.get("description")
                if not snippet:
                    snippets = result.get("snippets") or []
                    snippet = snippets[0] if snippets else ""
                output.append(
                    f"[{result.get('title', 'No Title')}]({result.get('url', '#')})\n{snippet}"
                )
            formatted = "\n\n".join(output) if output else "No results found."
        else:
            formatted = text or "No results found."
    except RequestException as e:
        return f"Error fetching You.com search: {e!s}"
    except Exception as e:
        return f"Error performing You.com search: {e!s}"
    return formatted
