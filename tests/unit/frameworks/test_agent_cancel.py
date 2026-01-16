"""Unit tests for AgentCancel exception class."""

from unittest.mock import patch

import pytest

from typing import Any

from any_agent import AgentCancel, AgentConfig, AgentFramework, AgentRunError, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID, LLM_IMPORT_PATHS
from any_agent.tracing.agent_trace import AgentTrace


class StopAgent(AgentCancel):
    """Test subclass of AgentCancel."""


class SpecificStopAgent(StopAgent):
    """Subclass of a subclass to test deep inheritance."""


class StopAgentRaiser(Callback):
    """Callback that raises StopAgent before any LLM interaction."""

    def before_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        msg = "Stopped before LLM"
        raise StopAgent(msg)


class RuntimeErrorRaiser(Callback):
    """Callback that raises RuntimeError before any LLM interaction (should be wrapped in AgentRunError)."""

    def before_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        msg = "Unexpected error"
        raise RuntimeError(msg)


class TestAgentCancel:
    """Tests for AgentCancel ABC."""

    def test_agent_cancel_is_abstract(self) -> None:
        """AgentCancel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            AgentCancel("cannot instantiate abstract class")

    def test_subclass_can_be_instantiated(self) -> None:
        """Subclasses of AgentCancel can be instantiated."""
        exc = StopAgent("test message")
        assert isinstance(exc, AgentCancel)
        assert isinstance(exc, Exception)

    def test_subclass_has_trace_property(self) -> None:
        """Subclasses have a trace property that is initially None."""
        exc = StopAgent("test")
        assert exc.trace is None

    def test_subclass_preserves_message(self) -> None:
        """Exception message is preserved."""
        exc = StopAgent("custom message")
        assert str(exc) == "custom message"

    def test_subclass_is_catchable_as_agent_cancel(self) -> None:
        """Subclasses can be caught as AgentCancel."""
        msg = "test"
        with pytest.raises(AgentCancel):
            raise StopAgent(msg)

    def test_subclass_is_catchable_as_exception(self) -> None:
        """Subclasses can be caught as Exception."""
        msg = "test"
        with pytest.raises(StopAgent, match="test"):
            raise StopAgent(msg)

    def test_subclass_is_catchable_by_specific_type(self) -> None:
        """Subclasses can be caught by their specific type."""
        msg = "test"
        with pytest.raises(StopAgent):
            raise StopAgent(msg)

    def test_trace_can_be_assigned(self) -> None:
        """Trace can be assigned (simulates framework behavior)."""
        exc = StopAgent("test")
        trace = AgentTrace()
        exc._trace = trace
        assert exc.trace is trace

    def test_deep_inheritance_works(self) -> None:
        """Subclass of a subclass can be instantiated and caught."""
        exc = SpecificStopAgent("deep")
        assert isinstance(exc, AgentCancel)
        assert isinstance(exc, StopAgent)

        msg = "test"
        with pytest.raises(AgentCancel):
            raise SpecificStopAgent(msg)

    def test_multiple_args_preserved(self) -> None:
        """Multiple exception arguments are preserved."""
        exc = StopAgent("arg1", "arg2")
        assert exc.args == ("arg1", "arg2")


class TestRunAsyncExceptionHandling:
    """Tests for AgentCancel handling in AnyAgent.run_async."""

    @pytest.mark.asyncio
    async def test_agent_cancel_propagates_without_wrapping(self) -> None:
        """AgentCancel raised in callback propagates directly, not wrapped."""
        agent_config = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            callbacks=[StopAgentRaiser()],
        )
        agent = await AnyAgent.create_async(AgentFramework.TINYAGENT, agent_config)

        # Patch LLM to ensure no external calls (callback raises first anyway).
        with (
            patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]),
            pytest.raises(StopAgent) as exc_info,
        ):
            await agent.run_async("test prompt")

        assert str(exc_info.value) == "Stopped before LLM"
        assert exc_info.value.trace is not None
        assert len(exc_info.value.trace.spans) > 0

    @pytest.mark.asyncio
    async def test_regular_exception_wrapped_in_agent_run_error(self) -> None:
        """Regular exceptions are wrapped in AgentRunError."""
        agent_config = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            callbacks=[RuntimeErrorRaiser()],
        )
        agent = await AnyAgent.create_async(AgentFramework.TINYAGENT, agent_config)

        # Patch LLM to ensure no external calls (callback raises first anyway).
        with (
            patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]),
            pytest.raises(AgentRunError) as exc_info,
        ):
            await agent.run_async("test prompt")

        assert isinstance(exc_info.value.original_exception, RuntimeError)
        assert str(exc_info.value.original_exception) == "Unexpected error"
        assert exc_info.value.trace is not None
        assert len(exc_info.value.trace.spans) > 0
