from typing import Any
from unittest.mock import AsyncMock, MagicMock

from any_agent import AgentFramework
from any_agent.callbacks.wrappers import _get_wrapper_by_framework
from any_agent.callbacks.wrappers.langchain import _LangChainWrapper


async def test_unwrap_before_wrap(agent_framework: AgentFramework) -> None:
    wrapper = _get_wrapper_by_framework(agent_framework)
    await wrapper.unwrap(MagicMock())


async def test_google_instrument_uninstrument() -> None:
    """Regression test for https://github.com/mozilla-ai/any-agent/issues/467"""
    agent = MagicMock()
    agent._agent.before_model_callback = None
    agent._agent.after_model_callback = None
    agent._agent.before_tool_callback = None
    agent._agent.after_tool_callback = None

    wrapper = _get_wrapper_by_framework(AgentFramework.GOOGLE)

    await wrapper.wrap(agent)
    assert callable(agent._agent.before_model_callback)
    assert callable(agent._agent.after_model_callback)
    assert callable(agent._agent.before_tool_callback)
    assert callable(agent._agent.after_tool_callback)

    await wrapper.unwrap(agent)
    assert agent._agent.before_model_callback is None
    assert agent._agent.after_model_callback is None
    assert agent._agent.before_tool_callback is None
    assert agent._agent.after_tool_callback is None


async def test_langchain_callback_raises_errors() -> None:
    """LangChain callback handler must have raise_error=True to propagate AgentCancel.

    By default, LangChain swallows exceptions in callback handlers and only logs
    warnings. Setting raise_error=True ensures exceptions (especially AgentCancel
    subclasses) propagate so they can be handled by run_async.
    """
    agent = MagicMock()
    agent._agent = MagicMock()
    agent._agent.ainvoke = AsyncMock()
    agent.config = MagicMock()
    agent.config.callbacks = []

    wrapper = _LangChainWrapper()
    await wrapper.wrap(agent)

    # Call the wrapped ainvoke to trigger callback injection.
    captured_kwargs: dict[str, Any] = {}

    async def capture_ainvoke(*args: Any, **kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock()

    # Replace the mock's original ainvoke to capture the kwargs.
    wrapper._original_ainvoke = capture_ainvoke
    await agent._agent.ainvoke("test")

    # Verify the callback was added with raise_error=True.
    assert "config" in captured_kwargs
    config = captured_kwargs["config"]
    # Config can be a dict or RunnableConfig, handle both.
    callbacks = (
        config.get("callbacks") if isinstance(config, dict) else config.callbacks
    )
    assert callbacks is not None
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert hasattr(callback, "raise_error")
    assert callback.raise_error is True
