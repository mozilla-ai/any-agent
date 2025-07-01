# mypy: disable-error-code="union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from any_agent.frameworks.google import GoogleAgent


class _GoogleADKWrapper:
    def __init__(self) -> None:
        self._original: dict[str, Any] = {}
        self.context: dict[str, Any] = {}

    async def wrap(self, agent: GoogleAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        self.context["tracer"] = agent._tracer
        self.context["running_traces"] = agent._running_traces

        self._original["before_model"] = agent._agent.before_model_callback

        def before_model_callback(
            callback_context: CallbackContext,
            llm_request: LlmRequest,
        ) -> Any | None:
            for callback in agent.config.callbacks:
                self.context = callback.before_llm_call(
                    self.context, callback_context, llm_request
                )

            if callable(self._original["before_model"]):
                return self._original["before_model"](callback_context, llm_request)

            return None

        agent._agent.before_model_callback = before_model_callback

        self._original["after_model"] = agent._agent.after_model_callback

        def after_model_callback(
            callback_context: CallbackContext,
            llm_response: LlmResponse,
        ) -> Any | None:
            for callback in agent.config.callbacks:
                self.context = callback.after_llm_call(
                    self.context, callback_context, llm_response
                )

            if callable(self._original["after_model"]):
                return self._original["after_model"](callback_context, llm_response)

            return None

        agent._agent.after_model_callback = after_model_callback

        self._original["before_tool"] = agent._agent.before_tool_callback

        def before_tool_callback(
            tool: BaseTool,
            args: dict[str, Any],
            tool_context: ToolContext,
        ) -> Any | None:
            for callback in agent.config.callbacks:
                self.context = callback.before_tool_execution(
                    self.context, tool, args, tool_context
                )

            if callable(self._original["before_tool"]):
                return self._original["before_tool"](tool, args, tool_context)

            return None

        agent._agent.before_tool_callback = before_tool_callback

        self._original["after_tool"] = agent._agent.after_tool_callback

        def after_tool_callback(
            tool: BaseTool,
            args: dict[str, Any],
            tool_context: ToolContext,
            tool_response: dict[Any, Any],
        ) -> Any | None:
            for callback in agent.config.callbacks:
                self.context = callback.after_tool_execution(
                    self.context, tool, args, tool_context, tool_response
                )

            if callable(self._original["after_tool"]):
                return self._original["after_tool"](
                    tool, args, tool_context, tool_response
                )

            return None

        agent._agent.after_tool_callback = after_tool_callback

    async def unwrap(self, agent: GoogleAgent) -> None:
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
