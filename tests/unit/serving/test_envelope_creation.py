import pytest
from pydantic import BaseModel

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a.types")

from a2a.types import TaskState

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.serving.a2a.envelope import (
    A2AEnvelope,
    _is_a2a_envelope,
    validate_a2a_output_type,
)


class CustomOutputType(BaseModel):
    custom_field: str
    result: str


class ValidA2AOutputType(A2AEnvelope[CustomOutputType]):
    pass


class MockAgent(AnyAgent):
    """Mock agent implementation for testing."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._agent = None

    async def _load_agent(self) -> None:
        pass

    async def _run_async(self, prompt: str, **kwargs: object) -> str:
        return "mock result"

    @property
    def framework(self) -> AgentFramework:
        from any_agent.config import AgentFramework

        return AgentFramework.TINYAGENT

    @classmethod
    def create(cls, framework: object, config: AgentConfig) -> "MockAgent":
        return cls(config)


@pytest.mark.asyncio
async def test_validation_passes_with_valid_a2a_envelope() -> None:
    """Test that validation passes when the agent's output_type properly inherits from A2AEnvelope."""
    # Create agent config with valid A2A envelope output_type
    config = AgentConfig(
        model_id="test-model", description="test agent", output_type=ValidA2AOutputType
    )

    # Create mock agent
    agent = MockAgent(config)

    # Validate agent for A2A - should not raise an exception
    validate_a2a_output_type(agent)

    assert agent.config.output_type is not None

    # Verify the output_type is a valid A2A envelope
    assert _is_a2a_envelope(agent.config.output_type)

    # Verify the envelope works correctly
    envelope_instance = agent.config.output_type(
        task_status=TaskState.completed,
        data=CustomOutputType(custom_field="test", result="custom result"),
    )

    assert isinstance(envelope_instance, A2AEnvelope)
    assert envelope_instance.task_status == TaskState.completed
    assert isinstance(envelope_instance.data, CustomOutputType)
    assert envelope_instance.data.custom_field == "test"
    assert envelope_instance.data.result == "custom result"


@pytest.mark.asyncio
async def test_validation_fails_without_a2a_envelope() -> None:
    """Test that validation fails when the agent's output_type doesn't inherit from A2AEnvelope."""
    # Create agent config with invalid output_type
    config = AgentConfig(
        model_id="test-model", description="test agent", output_type=CustomOutputType
    )

    # Create mock agent
    agent = MockAgent(config)

    # Validate agent for A2A - should raise ValueError
    with pytest.raises(
        ValueError, match="Agent output_type must inherit from A2AEnvelope"
    ):
        validate_a2a_output_type(agent)


@pytest.mark.asyncio
async def test_validation_fails_without_output_type() -> None:
    """Test that validation fails when the agent has no output_type."""
    # Create agent config without output_type
    config = AgentConfig(model_id="test-model", description="test agent")
    assert config.output_type is None

    # Create mock agent
    agent = MockAgent(config)

    # Validate agent for A2A - should raise ValueError
    with pytest.raises(
        ValueError, match="Agent output_type must inherit from A2AEnvelope"
    ):
        validate_a2a_output_type(agent)
