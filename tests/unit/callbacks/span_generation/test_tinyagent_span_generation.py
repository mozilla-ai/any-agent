import json
from unittest.mock import MagicMock

from any_agent.callbacks.span_generation.tinyagent import _TinyAgentSpanGeneration


def sample_tool_with_optional_params(
    keyphrase: str,
    license_type: str | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    is_official: bool = False,
) -> str:
    """Sample tool function with optional parameters for testing."""
    return f"keyphrase={keyphrase}, license={license_type}, categories={categories}, tags={tags}, is_official={is_official}"


def test_before_tool_execution_processes_arguments() -> None:
    """Test that before_tool_execution processes arguments and converts empty strings to None for optional parameters."""
    context = MagicMock()
    span_generation = _TinyAgentSpanGeneration()

    # modeled after https://github.com/mozilla-ai/any-agent/issues/802
    request = {
        "name": "sample_tool_with_optional_params",
        "arguments": {
            "keyphrase": "obsidian",
            "license_type": "",  # Should be converted to None
            "categories": "",  # Should be converted to None
            "tags": "",  # Should be converted to None
            "is_official": "false",  # Should be converted to False (boolean)
        },
    }

    span_generation.before_tool_execution(
        context, request, tool_function=sample_tool_with_optional_params
    )

    context.current_span.set_attributes.assert_called_once()
    call_args = context.current_span.set_attributes.call_args[0][0]

    processed_args = json.loads(call_args["gen_ai.tool.args"])

    assert processed_args["keyphrase"] == "obsidian"
    assert processed_args["license_type"] is None
    assert processed_args["categories"] is None
    assert processed_args["tags"] is None
    assert processed_args["is_official"] is False


def test_before_tool_execution_without_tool_function() -> None:
    """Test that before_tool_execution works without tool_function (fallback to raw args)."""
    context = MagicMock()
    span_generation = _TinyAgentSpanGeneration()

    request = {"name": "some_tool", "arguments": {"param1": "value1", "param2": ""}}

    span_generation.before_tool_execution(context, request)

    context.current_span.set_attributes.assert_called_once()
    call_args = context.current_span.set_attributes.call_args[0][0]

    processed_args = json.loads(call_args["gen_ai.tool.args"])

    assert processed_args["param1"] == "value1"
    assert processed_args["param2"] == ""
