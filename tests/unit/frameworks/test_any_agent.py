import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_create_any_with_framework(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(agent_framework, AgentConfig(model_id="gpt-4o"))
    assert agent


def test_create_any_with_valid_string(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(agent_framework.name, AgentConfig(model_id="gpt-4o"))
    assert agent


def test_create_any_with_invalid_string() -> None:
    with pytest.raises(ValueError, match="Unsupported agent framework"):
        AnyAgent.create("non-existing", AgentConfig(model_id="gpt-4o"))
