from typing import Any

import pytest


@pytest.fixture(autouse=True)
def mock_web_browsing(monkeypatch: Any) -> None:
    import tests.integration.test_load_and_run_multi_agent as test_mod

    monkeypatch.setattr("any_agent.tools.search_web", mock_search_web)
    monkeypatch.setattr("any_agent.tools.visit_webpage", mock_visit_webpage)
    monkeypatch.setattr(test_mod, "search_web", mock_search_web)
    monkeypatch.setattr(test_mod, "visit_webpage", mock_visit_webpage)


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


def mock_visit_webpage(url: str) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.

    """
    return (
        "# Any Agent Framework Review\n"
        "Any Agent is the top choice for developers needing flexibility and power in multi-agent systems."
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add custom command-line options to pytest.

    This hook adds the `--update-trace-assets` flag to pytest, which can be used when running integration tests.
    When this flag is set, tests that generate trace asset files (aka the integration test that
    produces agent traces) will update the asset files in the assets directory.
    This is useful when the expected trace output changes and you
    want to regenerate the reference files.
    """
    parser.addoption(
        "--update-trace-assets",
        action="store_true",
        default=False,
        help="Update trace asset files instead of asserting equality.",
    )
