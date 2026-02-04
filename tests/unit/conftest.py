import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from any_llm import AnyLLM

from any_agent.config import AgentFramework


@pytest.fixture(autouse=True)
def mock_verify_api_key() -> Generator[None, None, None]:
    """Mock AnyLLM._verify_and_set_api_key to skip API key validation in unit tests."""
    with patch("any_llm.AnyLLM._verify_and_set_api_key"):
        yield


@pytest.fixture(autouse=True)
def clear_any_llm_key() -> Generator[None, None, None]:
    """Clear ANY_LLM_KEY to prevent platform provider path in unit tests."""
    original = os.environ.pop(AnyLLM.ANY_LLM_KEY, None)
    yield
    if original is not None:
        os.environ[AnyLLM.ANY_LLM_KEY] = original


@pytest.fixture(autouse=True)
def mock_api_keys_for_unit_tests(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Automatically provide dummy API keys for unit tests to avoid API key requirements."""
    if "agent_framework" in request.fixturenames:
        agent_framework = request.getfixturevalue("agent_framework")
        # Only set dummy API key if we're in a test that uses the agent_framework fixture
        # and the framework is OPENAI (which uses any-llm with the AnyLLM.create class-based interface)
        if agent_framework == AgentFramework.OPENAI:
            os.environ["MISTRAL_API_KEY"] = "dummy-mistral-key-for-unit-tests"
    yield  # noqa: PT022
