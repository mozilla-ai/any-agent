import os

import pytest

from any_agent.tools import search_tavily


@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set; skipping Tavily integration test!",
)
def test_search_tavily_real() -> None:
    query = "Who is Leo Messi?"
    result = search_tavily(query)
    # Expect at least a title or content in the result
    assert "Leo Messi".lower() in result.lower() or "No results found." not in result
