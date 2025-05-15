import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock

from src.any_agent.tools import web_browsing


def test_search_tavily_no_client(monkeypatch):
    monkeypatch.setattr(web_browsing, "TavilyClient", None)
    result = web_browsing.search_tavily("test")
    assert "not installed" in result


def test_search_tavily_no_api_key(monkeypatch):
    monkeypatch.setattr(web_browsing, "TavilyClient", MagicMock())
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = web_browsing.search_tavily("test")
    assert "environment variable not set" in result


def test_search_tavily_success(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, query, include_images=False):
            return {
                "results": [
                    {
                        "title": "Test Title",
                        "url": "http://test.com",
                        "content": "Test content!",
                    }
                ]
            }

    monkeypatch.setattr(web_browsing, "TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = web_browsing.search_tavily("test")
    assert "Test Title" in result
    assert "Test content!" in result


def test_search_tavily_with_images(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, query, include_images=False):
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

    monkeypatch.setattr(web_browsing, "TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = web_browsing.search_tavily("test", include_images=True)
    assert "Test Title" in result
    assert "Images:" in result
    assert "cat.jpg" in result


def test_search_tavily_exception(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, query, include_images=False):
            raise RuntimeError("Oops!")

    monkeypatch.setattr(web_browsing, "TavilyClient", FakeClient)
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    result = web_browsing.search_tavily("test")
    assert "Error performing Tavily search" in result
