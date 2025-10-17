import os
from collections.abc import Generator

import pytest

from any_agent.config import AgentFramework


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
