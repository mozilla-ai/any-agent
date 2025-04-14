import json
from unittest.mock import patch, MagicMock

import pytest


from any_agent.config import AgentFramework, TracingConfig
from any_agent.tracing import RichConsoleSpanExporter


@pytest.fixture
def llm_span():
    class LLMSpan:
        def to_json(self):
            return json.dumps(
                {
                    "name": "ChatLiteLLM",
                    "context": {
                        "trace_id": "0x69ea1b41a9bc5724381993def669c803",
                        "span_id": "0x3817abcfb97cc40c",
                        "trace_state": "[]",
                    },
                    "kind": "SpanKind.INTERNAL",
                    "parent_id": "0x68ff5f13e03ac3fd",
                    "start_time": "2025-04-14T10:26:31.383147Z",
                    "end_time": "2025-04-14T10:26:32.563750Z",
                    "status": {"status_code": "OK"},
                    "attributes": {
                        "input.value": '{"messages": [[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "SystemMessage"], "kwargs": {"content": "Use the available tools to find the answer", "type": "system"}}, {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "Which agent framework is the best?", "type": "human", "id": "2aaf3de6-edf7-4cfa-9483-da348a6749da"}}]]}',
                        "input.mime_type": "application/json",
                        "output.value": '{"generations": [[{"text": "", "generation_info": {"finish_reason": "tool_calls"}, "type": "ChatGeneration", "message": {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "AIMessage"], "kwargs": {"content": "", "additional_kwargs": {"tool_calls": [{"lc": 1, "type": "not_implemented", "id": ["litellm", "types", "utils", "ChatCompletionMessageToolCall"], "repr": "ChatCompletionMessageToolCall(function=Function(arguments=\'{\\"query\\":\\"best agent frameworks 2023\\"}\', name=\'search_web\'), id=\'call_hYPpXiyNNLiRRytLlJdNFpGN\', type=\'function\')"}]}, "response_metadata": {"token_usage": {"lc": 1, "type": "not_implemented", "id": ["litellm", "types", "utils", "Usage"], "repr": "Usage(completion_tokens=20, prompt_tokens=145, total_tokens=165, completion_tokens_details=CompletionTokensDetailsWrapper(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=None), prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None))"}, "model": "gpt-4o-mini", "finish_reason": "tool_calls", "model_name": "gpt-4o-mini"}, "type": "ai", "id": "run-984943f2-6546-47fc-9b1d-81714109e374-0", "tool_calls": [{"name": "search_web", "args": {"query": "best agent frameworks 2023"}, "id": "call_hYPpXiyNNLiRRytLlJdNFpGN", "type": "tool_call"}], "usage_metadata": {"input_tokens": 145, "output_tokens": 20, "total_tokens": 165}, "invalid_tool_calls": []}}}]], "llm_output": {"token_usage": {"completion_tokens": 20, "prompt_tokens": 145, "total_tokens": 165, "completion_tokens_details": {"accepted_prediction_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0, "rejected_prediction_tokens": 0}, "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0}}, "model": "gpt-4o-mini"}, "run": null, "type": "LLMResult"}',
                        "output.mime_type": "application/json",
                        "llm.input_messages.0.message.role": "system",
                        "llm.input_messages.0.message.content": "Use the available tools to find the answer",
                        "llm.input_messages.1.message.role": "user",
                        "llm.input_messages.1.message.content": "Which agent framework is the best?",
                        "llm.output_messages.0.message.role": "assistant",
                        "llm.invocation_parameters": '{"model": "gpt-4o-mini", "temperature": null, "top_p": null, "top_k": null, "n": null, "_type": "litellm-chat", "stop": null, "tools": [{"type": "function", "function": {"name": "search_web", "description": "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.\\n\\n    Args:\\n        query (str): The search query to perform.\\n\\n    Returns:\\n        The top search results.", "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"}}}, {"type": "function", "function": {"name": "visit_webpage", "description": "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.\\n\\n    Args:\\n        url: The url of the webpage to visit.", "parameters": {"properties": {"url": {"type": "string"}}, "required": ["url"], "type": "object"}}}], "tool_choice": null}',
                        "llm.model_name": "gpt-4o-mini",
                        "llm.token_count.prompt": 145,
                        "llm.token_count.completion": 20,
                        "llm.token_count.total": 165,
                        "metadata": '{"langgraph_step": 1, "langgraph_node": "agent", "langgraph_triggers": ["branch:to:agent", "start:agent", "tools"], "langgraph_path": ["__pregel_pull", "agent"], "langgraph_checkpoint_ns": "agent:b43d5300-228d-6e29-4b2f-7994f7924340", "checkpoint_ns": "agent:b43d5300-228d-6e29-4b2f-7994f7924340", "ls_provider": "litellm", "ls_model_type": "chat", "ls_model_name": "gpt-4o-mini"}',
                        "openinference.span.kind": "LLM",
                    },
                    "events": [],
                    "links": [],
                    "resource": {
                        "attributes": {
                            "telemetry.sdk.language": "python",
                            "telemetry.sdk.name": "opentelemetry",
                            "telemetry.sdk.version": "1.32.0",
                            "service.name": "unknown_service",
                        },
                        "schema_url": "",
                    },
                }
            )

    return LLMSpan()


def test_rich_console_span_exporter_default(llm_span):
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(AgentFramework("langchain"), TracingConfig())
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_called()


def test_rich_console_span_exporter_disable(llm_span):
    console_mock = MagicMock()
    with patch("any_agent.tracing.Console", console_mock):
        exporter = RichConsoleSpanExporter(
            AgentFramework("langchain"), TracingConfig(llm=None)
        )
        exporter.export([llm_span])
        console_mock.return_value.rule.assert_not_called()
