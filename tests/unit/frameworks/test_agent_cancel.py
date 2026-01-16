"""Unit tests for AgentCancel exception class."""

from unittest.mock import patch

import pytest

from typing import Any

from any_agent import AgentCancel, AgentConfig, AgentFramework, AgentRunError, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.frameworks.any_agent import _unwrap_agent_cancel
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


class TestUnwrapAgentCancel:
    """Tests for _unwrap_agent_cancel helper function."""

    def test_returns_none_for_regular_exception(self) -> None:
        """Returns None when exception chain contains no AgentCancel."""
        exc = RuntimeError("regular error")
        assert _unwrap_agent_cancel(exc) is None

    def test_returns_none_for_chained_regular_exceptions(self) -> None:
        """Returns None when chained exceptions contain no AgentCancel."""
        inner = ValueError("inner")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        assert _unwrap_agent_cancel(outer) is None

    def test_finds_direct_agent_cancel(self) -> None:
        """Returns the exception itself if it is an AgentCancel."""
        exc = StopAgent("direct")
        result = _unwrap_agent_cancel(exc)
        assert result is exc

    def test_finds_agent_cancel_via_cause(self) -> None:
        """Finds AgentCancel in __cause__ (explicit raise from)."""
        cancel = StopAgent("wrapped")
        wrapper = RuntimeError("framework error")
        wrapper.__cause__ = cancel
        result = _unwrap_agent_cancel(wrapper)
        assert result is cancel

    def test_finds_agent_cancel_via_context(self) -> None:
        """Finds AgentCancel in __context__ (implicit chaining)."""
        cancel = StopAgent("wrapped")
        wrapper = RuntimeError("framework error")
        wrapper.__context__ = cancel
        result = _unwrap_agent_cancel(wrapper)
        assert result is cancel

    def test_finds_deeply_nested_agent_cancel(self) -> None:
        """Finds AgentCancel nested multiple levels deep."""
        cancel = StopAgent("deep")
        middle = ValueError("middle")
        middle.__cause__ = cancel
        outer = RuntimeError("outer")
        outer.__cause__ = middle
        result = _unwrap_agent_cancel(outer)
        assert result is cancel

    def test_prefers_cause_over_context(self) -> None:
        """When both __cause__ and __context__ exist, follows __cause__ first."""
        cause_cancel = StopAgent("from cause")
        context_cancel = SpecificStopAgent("from context")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause_cancel
        wrapper.__context__ = context_cancel
        result = _unwrap_agent_cancel(wrapper)
        assert result is cause_cancel

    def test_finds_subclass_of_agent_cancel(self) -> None:
        """Finds subclasses of AgentCancel (e.g., SpecificStopAgent)."""
        cancel = SpecificStopAgent("specific")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cancel
        result = _unwrap_agent_cancel(wrapper)
        assert result is cancel
        assert isinstance(result, StopAgent)
        assert isinstance(result, AgentCancel)
