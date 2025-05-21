# mypy: disable-error-code="no-untyped-call, no-untyped-def"
from __future__ import annotations

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
    from llama_index.core.instrumentation.events import BaseEvent
    from opentelemetry.trace import Tracer


class _LlamaIndexInstrumentor:
    def instrument(self, tracer: Tracer):
        class _AnyAgentEventHandler(BaseEventHandler):
            current_spans: dict[str, Span] = Field(default_factory=dict)

            @classmethod
            def class_name(cls) -> str:
                return "AnyAgentEventHandler"

            def handle(self, event: BaseEvent, **kwargs) -> Any:
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

                    self.current_spans[str(event.span_id)] = span
                elif isinstance(event, LLMChatEndEvent):
                    span = self.current_spans[str(event.span_id)]

                    if response := event.response:
                        span.set_attributes(
                            {
                                "genai.output": response.message.model_dump_json(
                                    exclude_none=True
                                ),
                                "genai.output.type": "json",
                            }
                        )
                        if token_usage := getattr(response, "raw", {}).get("usage"):
                            span.set_attributes(
                                {
                                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                                }
                            )
                    span.set_status(StatusCode.OK)
                    span.end()

        get_dispatcher().add_event_handler(_AnyAgentEventHandler())

    def uninstrument(self):
        dispatcher = get_dispatcher()
        dispatcher.event_handlers = [
            handler
            for handler in dispatcher.event_handlers
            if handler.class_name() != "AnyAgentEventHandler"
        ]
