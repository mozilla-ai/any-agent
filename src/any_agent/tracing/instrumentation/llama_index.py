from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from opentelemetry.trace import Span, StatusCode
from pydantic import Field

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.instrumentation.events import BaseEvent
    from opentelemetry.trace import Tracer


def _set_llm_input(messages: list[ChatMessage], span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.role.value,
                    "content": message.content or "No content",
                }
                for message in messages
            ]
        ),
    )


def _set_llm_output(response: ChatResponse, span: Span) -> None:
    if message := response.message:
        if content := message.content:
            span.set_attributes(
                {
                    "gen_ai.output": content,
                    "gen_ai.output.type": "text",
                }
            )
        if tool_calls := message.additional_kwargs.get("tool_calls", []):
            span.set_attributes(
                {
                    "gen_ai.output": json.dumps(
                        [
                            {
                                "tool.name": tool_call.get("function", {}).get(
                                    "name", "No name"
                                ),
                                "tool.args": tool_call.get("function", {}).get(
                                    "arguments", "No args"
                                ),
                            }
                            for tool_call in tool_calls
                        ],
                        default=str,
                        ensure_ascii=False,
                    ),
                    "gen_ai.output.type": "json",
                }
            )
    if raw := getattr(response, "raw", None):
        if token_usage := raw.get("usage"):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )


class _LlamaIndexInstrumentor:
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

    def instrument(self, tracer: Tracer) -> None:
        fist_llm_calls = self.first_llm_calls

        class _AnyAgentEventHandler(BaseEventHandler):
            current_spans: dict[str, Span] = Field(default_factory=dict)

            @classmethod
            def class_name(cls) -> str:
                return "AnyAgentEventHandler"

            def handle(self, event: BaseEvent, **kwargs) -> Any:  # type: ignore[no-untyped-def]
                if isinstance(event, LLMChatStartEvent):
                    model = event.model_dict["model"]
                    span: Span = tracer.start_span(
                        name=f"call_llm {model}",
                    )
                    span.set_attributes(
                        {
                            "gen_ai.operation.name": "call_llm",
                            "gen_ai.request.model": model,
                        }
                    )
                    trace_id = span.get_span_context().trace_id
                    if trace_id not in fist_llm_calls:
                        fist_llm_calls.add(trace_id)
                        _set_llm_input(event.messages, span)
                    self.current_spans[str(event.span_id)] = span

                elif isinstance(event, LLMChatEndEvent):
                    span = self.current_spans[str(event.span_id)]

                    if response := event.response:
                        _set_llm_output(response, span)

                    span.set_status(StatusCode.OK)
                    span.end()

        get_dispatcher().add_event_handler(_AnyAgentEventHandler())

    def uninstrument(self) -> None:
        dispatcher = get_dispatcher()
        dispatcher.event_handlers = [
            handler
            for handler in dispatcher.event_handlers
            if handler.class_name() != "AnyAgentEventHandler"
        ]
