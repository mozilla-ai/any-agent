import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from any_agent.callbacks.span_generation.llama_index import _LlamaIndexSpanGeneration
from any_agent.tracing.attributes import GenAI


def test_after_llm_call_records_tool_call_args() -> None:
    context = MagicMock()
    agent_output = SimpleNamespace(
        response=SimpleNamespace(content=""),
        tool_calls=[
            SimpleNamespace(
                tool_id="call_1",
                tool_name="search_web",
                tool_kwargs={"query": "weather in Paris"},
            )
        ],
        raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )

    _LlamaIndexSpanGeneration().after_llm_call(context, agent_output)

    attributes = context.current_span.set_attributes.call_args[0][0]
    assert json.loads(attributes[GenAI.OUTPUT]) == [
        {"tool.name": "search_web", "tool.args": {"query": "weather in Paris"}}
    ]
    assert attributes[GenAI.USAGE_INPUT_TOKENS] == 10
    assert attributes[GenAI.USAGE_OUTPUT_TOKENS] == 5
