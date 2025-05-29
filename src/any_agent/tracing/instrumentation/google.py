from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode
from wrapt.patches import resolve_path, wrap_object_attribute

from .common import _set_tool_output

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from opentelemetry.trace import Span, Tracer


def _set_llm_output(llm_response: LlmResponse, span: Span) -> None:
    content = llm_response.content
    if not content:
        return
    if not content.parts:
        return

    if content.parts[0].text:
        span.set_attributes(
            {
                "gen_ai.output": content.parts[0].text,
                "gen_ai.output.type": "text",
            }
        )
    else:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(part.function_call, "name", "No name"),
                            "tool.args": getattr(part.function_call, "args", "No args"),
                        }
                        for part in content.parts
                        if part.function_call
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )


def _set_llm_input(llm_request: LlmRequest, span: Span) -> None:
    if not llm_request.contents:
        return
    messages = []
    if config := llm_request.config:
        messages.append(
            {
                "role": "system",
                "content": getattr(config, "system_instruction", "No instructions"),
            }
        )
    if parts := llm_request.contents[0].parts:
        messages.append(
            {
                "role": getattr(llm_request.contents[0], "role", "No role"),
                "content": getattr(parts[0], "text", "No content"),
            }
        )
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(messages, default=str, ensure_ascii=False),
    )


class _GoogleADKTracingCallbacks:
    def __init__(self, tracer: Tracer) -> None:
        self._original_value = None
        self.tracer = tracer
        self._current_spans: dict[str, dict[str, Span]] = {
            "model": {},
            "tool": {},
        }
        self._original_callbacks: dict[str, Any] = {}
        self.first_llm_calls: set[int] = set()

    def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Any | None:
        span: Span = self.tracer.start_span(
            name=f"call_llm {llm_request.model}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": llm_request.model or "no_model",
            }
        )
        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(llm_request, span)
        self._current_spans["model"][callback_context.invocation_id] = span

        if callable(self._original_callbacks["LlmAgent.before_model_callback"]):
            return self._original_callbacks["LlmAgent.before_model_callback"](
                callback_context, llm_request
            )

        return None

    def before_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any | None:
        span: Span = self.tracer.start_span(
            name=f"execute_tool {tool.name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.description": tool.description,
                "gen_ai.tool.args": json.dumps(args),
                "gen_ai.tool.call.id": getattr(
                    tool_context, "function_call_id", "no_id"
                ),
            }
        )

        self._current_spans["tool"][tool_context.invocation_id] = span

        if callable(self._original_callbacks["LlmAgent.before_tool_callback"]):
            return self._original_callbacks["LlmAgent.before_tool_callback"](
                tool, args, tool_context
            )

        return None

    def after_model_callback(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Any | None:
        span = self._current_spans["model"][callback_context.invocation_id]

        _set_llm_output(llm_response, span)
        if resp_meta := llm_response.usage_metadata:
            if prompt_tokens := resp_meta.prompt_token_count:
                span.set_attributes({"gen_ai.usage.input_tokens": prompt_tokens})
            if output_tokens := resp_meta.candidates_token_count:
                span.set_attributes({"gen_ai.usage.output_tokens": output_tokens})
        span.set_status(StatusCode.OK)
        span.end()
        del self._current_spans["model"][callback_context.invocation_id]

        return None

    def after_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[Any, Any],
    ) -> Any | None:
        span = self._current_spans["tool"][tool_context.invocation_id]

        _set_tool_output(tool_response, span)

        span.set_status(StatusCode.OK)
        span.end()

        del self._current_spans["tool"][tool_context.invocation_id]

        if callable(self._original_callbacks["LlmAgent.before_tool_callback"]):
            return self._original_callbacks["LlmAgent.before_tool_callback"](
                tool, args, tool_context, tool_response
            )

        return None


class _GoogleADKInstrumentor:
    def __init__(self) -> None:
        self._original_callbacks: dict[str, Any] = {
            "LlmAgent.before_model_callback": None,
            "LlmAgent.before_tool_callback": None,
            "LlmAgent.after_model_callback": None,
            "LlmAgent.after_tool_callback": None,
        }

    def instrument(self, tracer: Tracer) -> None:
        callbacks = _GoogleADKTracingCallbacks(tracer=tracer)

        def callback_factory(value, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Honor any callback passed by the user
            kwargs["callbacks"]._original_callbacks[kwargs["name"]] = value
            self._original_callbacks[kwargs["name"]] = value
            return kwargs["callback_wrapper"]

        wrap_object_attribute(  # type: ignore[no-untyped-call]
            module="google.adk.agents.llm_agent",
            name="LlmAgent.before_model_callback",
            factory=callback_factory,
            kwargs={
                "name": "LlmAgent.before_model_callback",
                "callbacks": callbacks,
                "callback_wrapper": callbacks.before_model_callback,
            },
        )

        wrap_object_attribute(  # type: ignore[no-untyped-call]
            module="google.adk.agents.llm_agent",
            name="LlmAgent.before_tool_callback",
            factory=callback_factory,
            kwargs={
                "name": "LlmAgent.before_tool_callback",
                "callbacks": callbacks,
                "callback_wrapper": callbacks.before_tool_callback,
            },
        )

        wrap_object_attribute(  # type: ignore[no-untyped-call]
            module="google.adk.agents.llm_agent",
            name="LlmAgent.after_model_callback",
            factory=callback_factory,
            kwargs={
                "name": "LlmAgent.after_model_callback",
                "callbacks": callbacks,
                "callback_wrapper": callbacks.after_model_callback,
            },
        )

        wrap_object_attribute(  # type: ignore[no-untyped-call]
            module="google.adk.agents.llm_agent",
            name="LlmAgent.after_tool_callback",
            factory=callback_factory,
            kwargs={
                "name": "LlmAgent.after_tool_callback",
                "callbacks": callbacks,
                "callback_wrapper": callbacks.after_tool_callback,
            },
        )

    def uninstrument(self) -> None:
        module = "google.adk.agents.llm_agent"
        for name, original in self._original_callbacks.items():
            path, attribute = name.rsplit(".", 1)
            parent = resolve_path(module, path)[2]  # type: ignore[no-untyped-call]
            setattr(parent, attribute, original)
