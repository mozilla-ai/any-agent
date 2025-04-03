import pytest

from any_agent.instructions import get_instructions


def test_get_instructions_string():
    instructions = "You are a helpful assistant"
    result = get_instructions(instructions)
    assert result == instructions


def test_get_instructions_import():
    instructions = "agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX"
    result = get_instructions(instructions)
    assert result.startswith("# System context\nYou are part of a multi-agent system")


def test_get_instructions_import_error():
    instructions = "agents.extensions.fake_module.RECOMMENDED_PROMPT_PREFIX"
    with pytest.raises(ImportError):
        get_instructions(instructions)


def test_get_instructions_value_error():
    instructions = "any_agent.AnyAgent"
    with pytest.raises(ValueError, match="Instructions were identified as an import"):
        get_instructions(instructions)
