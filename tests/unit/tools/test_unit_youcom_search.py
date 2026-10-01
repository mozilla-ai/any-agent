import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import RequestException


def _response(payload: dict, session_id: str | None = None) -> Any:
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    if session_id:
        response.headers["Mcp-Session-Id"] = session_id
    response.json.return_value = payload
    return response


def test_search_youcom_success(monkeypatch: Any) -> None:
    search_payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [
                            {
                                "title": "Test Title",
                                "url": "http://test.com",
                                "description": "Test content!",
                            }
                        ]
                    ),
                }
            ]
        }
    }
    responses = [
        _response({"result": {}}, session_id="session-1"),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)) as mock_post:
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "Test Title" in result
        assert "Test content!" in result
        # initialize + initialized notification + tools/call
        assert mock_post.call_count == 3
        assert "https://api.you.com/mcp?profile=free" in str(mock_post.call_args)
        # session id from the first response must be passed along
        assert mock_post.call_args.kwargs["headers"]["Mcp-Session-Id"] == "session-1"


def test_search_youcom_no_results(monkeypatch: Any) -> None:
    search_payload = {"result": {"content": [{"type": "text", "text": json.dumps([])}]}}
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert result == "No results found."


def test_search_youcom_request_exception(monkeypatch: Any) -> None:
    with patch(
        "requests.post", MagicMock(side_effect=RequestException("network down"))
    ):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "Error fetching You.com search" in result


def test_search_youcom_authenticated_endpoint(monkeypatch: Any) -> None:
    search_payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [{"title": "T", "url": "http://t.co", "description": "D"}]
                    ),
                }
            ]
        }
    }
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)) as mock_post:
        monkeypatch.setenv("YDC_API_KEY", "fake-key")
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "T" in result
        args = mock_post.call_args
        assert str(args.args[0]) == "https://api.you.com/mcp"
        assert args.kwargs["headers"]["Authorization"] == "Bearer fake-key"


def test_search_youcom_non_json_text(monkeypatch: Any) -> None:
    search_payload = {
        "result": {"content": [{"type": "text", "text": "plain fallback text"}]}
    }
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert result == "plain fallback text"


def test_search_youcom_snippets_fallback(monkeypatch: Any) -> None:
    search_payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [
                            {
                                "title": "Test Title",
                                "url": "http://test.com",
                                "snippets": ["Snippet text"],
                            }
                        ]
                    ),
                }
            ]
        }
    }
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "Test Title" in result
        assert "Snippet text" in result


@pytest.mark.parametrize(
    "query",
    ["what is agent eval", "latest open source LLM news"],
)
def test_search_youcom_query_forwarded(monkeypatch: Any, query: str) -> None:
    search_payload = {"result": {"content": [{"type": "text", "text": json.dumps([])}]}}
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)) as mock_post:
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        search_youcom(query)
        call = mock_post.call_args.kwargs["json"]
        assert call["params"]["arguments"]["query"] == query
        assert call["params"]["name"] == "you-search"


def test_search_youcom_handles_empty_notification_response(monkeypatch: Any) -> None:
    accepted = MagicMock()
    accepted.status_code = 202
    accepted.headers = {"Content-Type": "application/json"}
    accepted.text = ""

    search_payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [
                            {
                                "title": "Test Title",
                                "url": "http://test.com",
                                "description": "Test content!",
                            }
                        ]
                    ),
                }
            ]
        }
    }
    responses = [
        _response({}),
        accepted,
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "Test Title" in result


def test_search_youcom_nested_results_shape(monkeypatch: Any) -> None:
    search_payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": {
                                "web": [
                                    {
                                        "title": "Nested Title",
                                        "url": "http://nested.com",
                                        "description": "Nested content!",
                                    }
                                ]
                            }
                        }
                    ),
                }
            ]
        }
    }
    responses = [
        _response({}),
        _response({}),
        _response(search_payload),
    ]
    with patch("requests.post", MagicMock(side_effect=responses)):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        from any_agent.tools import search_youcom

        result = search_youcom("test")
        assert "Nested Title" in result
        assert "Nested content!" in result
        assert "[Nested Title](http://nested.com)" in result
