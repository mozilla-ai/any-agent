import pytest

from any_agent import AgentConfig, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.config import AgentFramework
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID
from typing import Any


class LLMInputModifier(Callback):
    """Callback that modifies LLM input messages."""

    def __init__(self) -> None:
        self.original_messages: list[dict[str, Any]] = []
        self.modified_messages: list[dict[str, Any]] = []

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        # Capture original messages for verification
        messages = context.framework_state.get_messages()
        self.original_messages = [msg.copy() for msg in messages]

        # Verify message structure before modification
        assert len(messages) > 0, "Expected at least one message"
        assert "role" in messages[-1], "Expected 'role' key in message"
        assert "content" in messages[-1], "Expected 'content' key in message"

        # Modify the last message
        messages[-1]["content"] = "Say hello"
        context.framework_state.set_messages(messages)

        # Capture modified messages for verification
        self.modified_messages = context.framework_state.get_messages()

        return context


class SecondModifier(Callback):
    """Second callback to test sequential modifications."""

    def __init__(self) -> None:
        self.saw_first_modification = False

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        messages = context.framework_state.get_messages()

        # Verify the first callback's modification is present
        if len(messages) > 0:
            self.saw_first_modification = "Say hello" in messages[-1].get("content", "")

        # Add additional modification
        if len(messages) > 0:
            messages[-1]["content"] = "Say hello and goodbye"
            context.framework_state.set_messages(messages)

        return context


async def test_modify_llm_input(agent_framework: AgentFramework) -> None:
    """Test that framework_state message modification works via helper methods."""
    modifier = LLMInputModifier()
    second_modifier = SecondModifier()

    config = AgentConfig(
        model_id="openai:gpt-4.1-mini",
        instructions="You are a helpful assistant.",
        callbacks=[modifier, second_modifier],
    )

    agent = await AnyAgent.create_async(agent_framework, config)

    try:
        # First run: Test modification and sequential callback behavior
        result = await agent.run_async("Say goodbye")
        assert result.final_output is not None
        assert isinstance(result.final_output, str)

        # Verify the modification took effect (should say hello AND goodbye)
        assert "hello" in result.final_output.lower(), (
            "Expected 'hello' in the final output from first modification"
        )
        assert "goodbye" in result.final_output.lower(), (
            "Expected 'goodbye' in the final output from second modification"
        )

        # Verify we captured the original message before modification
        assert len(modifier.original_messages) > 0, (
            "Should have captured original messages"
        )
        assert "Say goodbye" in modifier.original_messages[-1]["content"], (
            "Original message should contain 'Say goodbye'"
        )

        # Verify message structure was preserved
        assert "role" in modifier.modified_messages[-1], (
            "Modified message should have 'role' key"
        )
        assert "content" in modifier.modified_messages[-1], (
            "Modified message should have 'content' key"
        )

        # Verify sequential callback execution
        assert second_modifier.saw_first_modification, (
            "Second callback should see first callback's modification"
        )

    finally:
        await agent.cleanup_async()
