from any_agent import AgentConfig, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.config import AgentFramework
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID
from typing import Any


class LLMInputModifier(Callback):
    """Callback that modifies LLM input messages."""

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        messages = context.framework_state.get_messages()
        messages[-1]["content"] = "Say hello"
        context.framework_state.set_messages(messages)
        return context


async def test_modify_llm_input(agent_framework: AgentFramework) -> None:
    """Test that framework_state message modification works via helper methods."""
    modifier = LLMInputModifier()
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions="You are a helpful assistant.",
        callbacks=[modifier],
    )

    agent = await AnyAgent.create_async(agent_framework, config)

    try:
        result = await agent.run_async("Say goodbye")
        assert result.final_output is not None
        assert isinstance(result.final_output, str)

        assert "hello" in result.final_output.lower(), (
            "Expected 'hello' in the final output"
        )

    finally:
        await agent.cleanup_async()
