from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from any_llm import acompletion, completion

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    from agno.agent import Agent
    from agno.models.base import Model
    from agno.models.response import ModelResponse
    from agno.tools.toolkit import Toolkit

    agno_available = True
except ImportError:
    agno_available = False


if TYPE_CHECKING:
    from agno.agent import RunResponse
    from agno.models.message import Message
    from pydantic import BaseModel


@dataclass
class AnyLLM(Model):
    """A class for interacting with any-llm.

    any-llm allows you to use a unified interface for various LLM providers.
    For more information, see: https://mozilla-ai.github.io/any-llm/
    """

    id: str = "gpt-4o"
    name: str = "any-llm"
    provider: str = "any-llm"

    api_key: str | None = None
    api_base: str | None = None

    request_params: dict[str, Any] | None = None

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for any-llm API."""
        formatted_messages = []
        for m in messages:
            msg = {
                "role": m.role,
                "content": m.content if m.content is not None else "",
            }

            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]

            if m.role == "tool":
                msg["tool_call_id"] = m.tool_call_id or ""
                msg["name"] = m.name or ""

            formatted_messages.append(msg)

        return formatted_messages

    def get_request_params(
        self, tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Return keyword arguments for API requests.

        Returns:
            Dict[str, Any]: The API kwargs for the model.

        """
        request_params: dict[str, Any] = {
            "model": self.id,
            "api_base": self.api_base,
            "api_key": self.api_key,
            **self.request_params,  # type: ignore[dict-item]
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        return request_params

    def invoke(
        self,
        messages: list[Message],
        response_format: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        completion_kwargs = self.get_request_params(tools=tools)
        completion_kwargs["messages"] = self._format_messages(messages)
        return completion(**completion_kwargs)

    async def ainvoke(
        self,
        messages: list[Message],
        response_format: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        completion_kwargs = self.get_request_params(tools=tools)
        completion_kwargs["messages"] = self._format_messages(messages)
        return await acompletion(**completion_kwargs)

    def invoke_stream(
        self,
        messages: list[Message],
        response_format: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        completion_kwargs = self.get_request_params(tools=tools)
        completion_kwargs["messages"] = self._format_messages(messages)
        completion_kwargs["stream"] = True
        return completion(**completion_kwargs)

    async def ainvoke_stream(
        self,
        messages: list[Message],
        response_format: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        completion_kwargs = self.get_request_params(tools=tools)
        completion_kwargs["messages"] = self._format_messages(messages)
        completion_kwargs["stream"] = True
        return acompletion(**completion_kwargs)

    def parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:  # type: ignore[no-untyped-def]
        """Parse the provider response."""
        model_response = ModelResponse()

        response_message = response.choices[0].message

        if response_message.content is not None:
            model_response.content = response_message.content

        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            model_response.tool_calls = []
            for tool_call in response_message.tool_calls:
                model_response.tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        if response.usage is not None:
            model_response.response_usage = response.usage

        return model_response

    def parse_provider_response_delta(self, response_delta: Any) -> ModelResponse:
        """Parse the provider response delta for streaming responses."""
        model_response = ModelResponse()

        if hasattr(response_delta, "choices") and len(response_delta.choices) > 0:
            choice_delta = response_delta.choices[0].delta

            if choice_delta:
                if (
                    hasattr(choice_delta, "content")
                    and choice_delta.content is not None
                ):
                    model_response.content = choice_delta.content

                if hasattr(choice_delta, "tool_calls") and choice_delta.tool_calls:
                    processed_tool_calls = []
                    for tool_call in choice_delta.tool_calls:
                        actual_index = (
                            getattr(tool_call, "index", 0)
                            if hasattr(tool_call, "index")
                            else 0
                        )

                        tool_call_dict = {"index": actual_index, "type": "function"}

                        if hasattr(tool_call, "id") and tool_call.id is not None:
                            tool_call_dict["id"] = tool_call.id

                        function_data = {}
                        if hasattr(tool_call, "function"):
                            if (
                                hasattr(tool_call.function, "name")
                                and tool_call.function.name is not None
                            ):
                                function_data["name"] = tool_call.function.name
                            if (
                                hasattr(tool_call.function, "arguments")
                                and tool_call.function.arguments is not None
                            ):
                                function_data["arguments"] = (
                                    tool_call.function.arguments
                                )

                        tool_call_dict["function"] = function_data
                        processed_tool_calls.append(tool_call_dict)

                    model_response.tool_calls = processed_tool_calls

        if hasattr(response_delta, "usage") and response_delta.usage is not None:
            model_response.response_usage = response_delta.usage

        return model_response


DEFAULT_MODEL_TYPE = AnyLLM


class AgnoAgent(AnyAgent):
    """Agno agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: Agent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.AGNO

    def _get_model(self, agent_config: AgentConfig) -> Model:
        """Get the model configuration for an Agno agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE

        return model_type(
            id=agent_config.model_id,
            api_base=agent_config.api_base,
            api_key=agent_config.api_key,
            request_params=agent_config.model_args or {},  # type: ignore[arg-type]
        )

    @staticmethod
    def _unpack_tools(tools: list[Any]) -> list[Any]:
        unpacked: list[Any] = []
        for tool in tools:
            if isinstance(tool, Toolkit):
                unpacked.extend(f for f in tool.functions.values())
            else:
                unpacked.append(tool)
        return unpacked

    async def _load_agent(self) -> None:
        if not agno_available:
            msg = "You need to `pip install 'any-agent[agno]'` to use this agent"
            raise ImportError(msg)

        tools = await self._load_tools(self.config.tools)

        self._tools = self._unpack_tools(tools)

        agent_args = self.config.agent_args or {}
        if self.config.output_type:
            agent_args["response_model"] = self.config.output_type
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            tools=tools,
            **agent_args,
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result: RunResponse = await self._agent.arun(prompt, **kwargs)
        return result.content  # type: ignore[return-value]

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, we need to recreate it with the new output type
        # The AGNO agent requires response_model to be set during construction
        if self._agent:
            # Rebuild tools list from original config
            tools = await self._load_tools(self.config.tools)

            # Recreate the agent with the new configuration
            agent_args = self.config.agent_args or {}
            if output_type:
                agent_args["response_model"] = output_type

            self._agent = Agent(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self._get_model(self.config),
                tools=tools,
                **agent_args,
            )

            # Update the tools list
            self._tools = self._unpack_tools(tools)
