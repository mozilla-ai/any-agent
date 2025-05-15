from typing import Any
from unittest.mock import MagicMock

from any_agent.tools import search_tavily


def test_search_tavily_no_client(monkeypatch: Any) -> None:
    monkeypatch.setattr("any_agent.tools.web_browsing.TavilyClient", None)
    result = search_tavily("test")
    assert "not installed" in result


def test_search_tavily_no_api_key(monkeypatch: Any) -> None:
    monkeypatch.setattr("any_agent.tools.web_browsing.TavilyClient", MagicMock())
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = search_tavily("test")
    assert "environment variable not set" in result


def test_search_tavily_success(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            return {
                "results": [
                    {
                        "title": "Test Title",
                        "url": "http://test.com",
                        "content": "Test content!",
                    }
                ]
            }

    monkeypatch.setattr("any_agent.tools.web_browsing.TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = search_tavily("test")
    assert "Test Title" in result
    assert "Test content!" in result


def test_search_tavily_with_images(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            return {
                "results": [
                    {
                        "title": "Test Title",
                        "url": "http://test.com",
                        "content": "Test content!",
                    }
                ],
                "images": ["http://image.com/cat.jpg"],
            }

    monkeypatch.setattr("any_agent.tools.web_browsing.TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = search_tavily("test", include_images=True)
    assert "Test Title" in result
    assert "Images:" in result
    assert "cat.jpg" in result


def test_search_tavily_exception(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            msg = "Oops!"
            raise RuntimeError(msg)

    monkeypatch.setattr("any_agent.tools.web_browsing.TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = search_tavily("test")
    assert "Error performing Tavily search" in result
