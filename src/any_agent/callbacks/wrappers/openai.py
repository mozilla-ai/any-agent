# mypy: disable-error-code="misc,method-assign,no-untyped-def,no-untyped-call,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.openai import OpenAIAgent


class _OpenAIAgentsWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_llm_call: Callable[..., Any] | None = None
        self._original_get_all_tools: Callable[..., Any] | None = None
        self._original_invokes: dict[str, Any] = {}

    async def wrap(self, agent: OpenAIAgent) -> None:
        self._original_llm_call = agent._agent.model.get_response

        async def wrapped_llm_call(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = getattr(agent._agent.model, "model", None)

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            output = await self._original_llm_call(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(
                    context,
                    output,
                )

            return output

        agent._agent.model.get_response = wrapped_llm_call

        # OpenAI dynamically converts the MCP tools into FunctionTools during run.
        self._original_get_all_tools = agent._agent.get_all_tools

        async def wrapped_tool_execution(
            original_tool, original_invoke, *args, **kwargs
        ):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["original_tool"] = original_tool

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            output = await original_invoke(*args, **kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(
                    context,
                    output,
                )

            return output

        async def wrapped_get_all_tools(run_context):
            all_tools = await self._original_get_all_tools(run_context)

            class WrappedTool:
                def __init__(self, original_tool, original_invoke):
                    self.original_tool = original_tool
                    self.original_invoke = original_invoke

                async def on_invoke_tool(self, *args, **kwargs):
                    return await wrapped_tool_execution(
                        self.original_tool, self.original_invoke, *args, **kwargs
                    )

            for original_tool in all_tools:
                original_invoke = original_tool.on_invoke_tool
                self._original_invokes[original_tool.name] = original_invoke

                wrapped_tool = WrappedTool(original_tool, original_invoke)
                original_tool.on_invoke_tool = wrapped_tool.on_invoke_tool

            return all_tools

        agent._agent.get_all_tools = wrapped_get_all_tools

    async def unwrap(self, agent: OpenAIAgent) -> None:
        if self._original_llm_call is not None:
            agent._agent.model.get_response = self._original_llm_call

        if self._original_get_all_tools is not None:
            agent._agent.get_all_tools = self._original_get_all_tools

        # Only non-mcp tools need to be unwrapped.
        for tool in agent._agent.tools:
            if tool.name in self._original_invokes:
                tool.on_invoke_tool = self._original_invokes[tool.name]
