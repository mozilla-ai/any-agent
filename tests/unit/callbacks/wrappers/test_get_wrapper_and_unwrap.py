from unittest.mock import MagicMock

from any_agent import AgentFramework
from any_agent.callbacks.wrappers import _get_wrapper_by_framework


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
