from any_agent import AgentConfig, AnyAgent
from any_agent.callbacks import Callback, Context
from any_agent.config import AgentFramework
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID
from typing import Any


class LLMInputModifier(Callback):
    """Callback that modifies LLM input messages."""

    def __init__(self):
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

    def __init__(self):
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


class AfterLLMModifier(Callback):
    """Callback that modifies messages after LLM response."""

    def __init__(self):
        self.llm_response_messages: list[dict[str, Any]] = []
        self.saw_llm_response = False

    def after_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        # Get messages including the LLM's response
        messages = context.framework_state.get_messages()
        self.llm_response_messages = [msg.copy() for msg in messages]

        # Verify we can see the LLM's response
        if len(messages) > 0:
            last_msg = messages[-1]
            # The LLM should have responded with hello and goodbye
            self.saw_llm_response = (
                "hello" in last_msg.get("content", "").lower() and
                "goodbye" in last_msg.get("content", "").lower()
            )

            # Modify the LLM's response by appending text
            last_msg["content"] = last_msg["content"] + " Also, have a great day!"
            context.framework_state.set_messages(messages)

        return context


async def test_modify_llm_input(agent_framework: AgentFramework) -> None:
    """Test that framework_state message modification works via helper methods.

    This test verifies:
    1. Original messages can be read before modification (before_llm_call)
    2. Message structure (role, content) is preserved after modifications
    3. Multiple callbacks can modify messages sequentially (before_llm_call)
    4. LLM response can be read and modified in after_llm_call
    5. Modifications don't leak between separate runs (isolation)
    """
    modifier = LLMInputModifier()
    second_modifier = SecondModifier()
    after_llm_modifier = AfterLLMModifier()

    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions="You are a helpful assistant.",
        callbacks=[modifier, second_modifier, after_llm_modifier],
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

        # Verify after_llm_call modification took effect
        assert "have a great day" in result.final_output.lower(), (
            "Expected 'have a great day' from after_llm_call modification"
        )

        # Verify we captured the original message before modification
        assert len(modifier.original_messages) > 0, "Should have captured original messages"
        assert "Say goodbye" in modifier.original_messages[-1]["content"], (
            "Original message should contain 'Say goodbye'"
        )

        # Verify message structure was preserved
        assert "role" in modifier.modified_messages[-1], "Modified message should have 'role' key"
        assert "content" in modifier.modified_messages[-1], "Modified message should have 'content' key"

        # Verify sequential callback execution
        assert second_modifier.saw_first_modification, (
            "Second callback should see first callback's modification"
        )

        # Verify after_llm_call could read the LLM's response
        assert after_llm_modifier.saw_llm_response, (
            "after_llm_call callback should be able to read LLM response"
        )
        assert len(after_llm_modifier.llm_response_messages) > 0, (
            "after_llm_call should have captured messages including LLM response"
        )

        # Second run: Test that modifications don't leak between runs
        result2 = await agent.run_async("Tell me a joke")
        assert result2.final_output is not None
        assert isinstance(result2.final_output, str)

        # Verify the second run also got modified correctly
        # (should still say hello and goodbye, not "tell me a joke")
        assert "hello" in result2.final_output.lower(), (
            "Expected 'hello' in second run output"
        )

        # Verify original message was different for second run
        assert "Tell me a joke" in modifier.original_messages[-1]["content"], (
            "Second run should have different original message"
        )
        assert "Tell me a joke" not in modifier.modified_messages[-1]["content"], (
            "Second run's original input should have been modified"
        )

    finally:
        await agent.cleanup_async()
