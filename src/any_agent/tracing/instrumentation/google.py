# mypy: disable-error-code="union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from opentelemetry.trace import Span

    from any_agent.frameworks.google import GoogleAgent


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


class _GoogleADKInstrumentor:
    def __init__(self) -> None:
        self._original: dict[str, Any] = {}
        self.first_llm_calls: set[int] = set()
        self._current_spans: dict[str, dict[str, Span]] = {
            "model": {},
            "tool": {},
        }

    def instrument(self, agent: GoogleAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        tracer = agent._tracer

        self._original["before_model"] = agent._agent.before_model_callback

        def before_model_callback(
            callback_context: CallbackContext,
            llm_request: LlmRequest,
        ) -> Any | None:
            span: Span = tracer.start_span(
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

            if callable(self._original["before_model"]):
                return self._original["before_model"](callback_context, llm_request)

            return None

        agent._agent.before_model_callback = before_model_callback

        self._original["before_tool"] = agent._agent.before_tool_callback

        def before_tool_callback(
            tool: BaseTool,
            args: dict[str, Any],
            tool_context: ToolContext,
        ) -> Any | None:
            span: Span = tracer.start_span(
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

            if callable(self._original["before_tool"]):
                return self._original["before_tool"](tool, args, tool_context)

            return None

        agent._agent.before_tool_callback = before_tool_callback

        self._original["after_model"] = agent._agent.after_model_callback

        def after_model_callback(
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
            trace_id = span.get_span_context().trace_id
            agent._running_traces[trace_id].add_span(span)

            del self._current_spans["model"][callback_context.invocation_id]

            if callable(self._original["after_model"]):
                return self._original["after_model"](callback_context, llm_response)

            return None

        agent._agent.after_model_callback = after_model_callback

        self._original["after_tool"] = agent._agent.after_tool_callback

        def after_tool_callback(
            tool: BaseTool,
            args: dict[str, Any],
            tool_context: ToolContext,
            tool_response: dict[Any, Any],
        ) -> Any | None:
            span = self._current_spans["tool"][tool_context.invocation_id]

            _set_tool_output(tool_response, span)
            span.end()

            trace_id = span.get_span_context().trace_id
            agent._running_traces[trace_id].add_span(span)

            del self._current_spans["tool"][tool_context.invocation_id]

            if callable(self._original["after_tool"]):
                return self._original["after_tool"](
                    tool, args, tool_context, tool_response
                )

            return None

        agent._agent.after_tool_callback = after_tool_callback

    def uninstrument(self, agent: GoogleAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        if "before_model" in self._original:
            agent._agent.before_model_callback = self._original["before_model"]
        if "before_tool" in self._original:
            agent._agent.before_tool_callback = self._original["before_tool"]
        if "after_model" in self._original:
            agent._agent.after_model_callback = self._original["after_model"]
        if "after_tool" in self._original:
            agent._agent.after_tool_callback = self._original["after_tool"]
