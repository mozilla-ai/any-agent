import pytest
from pydantic import BaseModel

# Skip entire module if a2a dependencies are not available
pytest.importorskip("a2a.types")
pytest.importorskip("any_agent.serving.envelope")

from a2a.types import TaskState

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.serving.a2a.envelope import (
    A2AEnvelope,
    _DefaultBody,
    _is_a2a_envelope,
    prepare_agent_for_a2a_async,
)


class CustomOutputType(BaseModel):
    custom_field: str
    result: str


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
async def test_envelope_created_without_output_type() -> None:
    """Test that the envelope is correctly created when the agent is configured without an output_type."""
    # Create agent config without output_type
    config = AgentConfig(model_id="test-model", description="test agent")
    assert config.output_type is None

    # Create mock agent
    agent = MockAgent(config)

    # Prepare agent for A2A
    prepared_agent = await prepare_agent_for_a2a_async(agent)

    # Verify the envelope was created with default body
    assert prepared_agent.config.output_type is not None
    assert _is_a2a_envelope(prepared_agent.config.output_type)

    # Verify the envelope wraps _DefaultBody
    envelope_instance = prepared_agent.config.output_type(
        task_status=TaskState.completed, data=_DefaultBody(result="test result")
    )

    assert isinstance(envelope_instance, A2AEnvelope)
    assert envelope_instance.task_status == TaskState.completed
    assert isinstance(envelope_instance.data, _DefaultBody)
    assert envelope_instance.data.result == "test result"


@pytest.mark.asyncio
async def test_envelope_created_with_output_type() -> None:
    """Test that the envelope is correctly created when an agent is configured with an output_type."""
    # Create agent config with custom output_type
    config = AgentConfig(
        model_id="test-model", description="test agent", output_type=CustomOutputType
    )

    # Create mock agent
    agent = MockAgent(config)

    # Prepare agent for A2A
    prepared_agent = await prepare_agent_for_a2a_async(agent)

    # Verify the envelope was created with custom output type
    assert prepared_agent.config.output_type is not None
    assert _is_a2a_envelope(prepared_agent.config.output_type)

    # Verify the envelope wraps the custom output type
    envelope_instance = prepared_agent.config.output_type(
        task_status=TaskState.completed,
        data=CustomOutputType(custom_field="test", result="custom result"),
    )

    assert isinstance(envelope_instance, A2AEnvelope)
    assert envelope_instance.task_status == TaskState.completed
    assert isinstance(envelope_instance.data, CustomOutputType)
    assert envelope_instance.data.custom_field == "test"
    assert envelope_instance.data.result == "custom result"
