from __future__ import annotations

import math
import time
from copy import copy
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

try:
    import litellm
    from agents import Agent, Model, ModelSettings, Runner, Tool, Usage
    from agents.extensions.models.litellm_model import LitellmConverter
    from agents.items import ModelResponse
    from agents.models.chatcmpl_converter import Converter
    from agents.models.chatcmpl_helpers import HEADERS
    from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler
    from agents.models.fake_id import FAKE_RESPONSES_ID
    from agents.tracing import generation_span
    from agents.util._json import _to_dump_compatible
    from openai import NOT_GIVEN, NotGiven
    from openai.types.responses import Response
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
    )

    agents_available = True
except ImportError:
    agents_available = False

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import TResponseInputItem, TResponseStreamEvent
    from agents.models.interface import ModelTracing
    from agents.tracing.span_data import GenerationSpanData
    from agents.tracing.spans import Span
    from openai import AsyncStream
    from openai.types.chat import ChatCompletionChunk
    from pydantic import BaseModel


class LitellmModel(Model):
    """Enables using any model via LiteLLM.

    LiteLLM allows you to access OpenAPI, Anthropic, Gemini, Mistral, and many other models.
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
            )

            assert isinstance(response.choices[0], litellm.types.utils.Choices)

            usage = Usage()
            if hasattr(response, "usage"):
                response_usage = response.usage
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response_usage.prompt_tokens,
                        output_tokens=response_usage.completion_tokens,
                        total_tokens=response_usage.total_tokens,
                        input_tokens_details=InputTokensDetails(
                            cached_tokens=getattr(
                                response_usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        ),
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=getattr(
                                response_usage.completion_tokens_details,
                                "reasoning_tokens",
                                0,
                            )
                            or 0
                        ),
                    )
                    if response.usage
                    else Usage()
                )

            if tracing.include_data():
                span_generation.span_data.output = [
                    response.choices[0].message.model_dump()
                ]
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = Converter.message_to_output_items(
                LitellmConverter.convert_message_to_openai(response.choices[0].message)
            )

            return ModelResponse(
                output=items,
                usage=usage,
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response, stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
                prompt=prompt,
            )

            final_response: Response | None = None
            async for chunk in ChatCmplStreamHandler.handle_stream(response, stream):
                yield chunk

                if chunk.type == "response.completed":
                    final_response = chunk.response

            if tracing.include_data() and final_response:
                span_generation.span_data.output = [final_response.model_dump()]

            if final_response and final_response.usage:
                span_generation.span_data.usage = {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
        prompt: Any | None = None,
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: Any | None = None,
    ) -> (
        litellm.types.utils.ModelResponse
        | tuple[Response, AsyncStream[ChatCompletionChunk]]
    ):
        converted_messages = Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = (
            [Converter.tool_to_openai(tool) for tool in tools] if tools else []
        )

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)

        reasoning_effort = (
            model_settings.reasoning.effort if model_settings.reasoning else None
        )

        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        extra_kwargs = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = copy(model_settings.extra_query)
        if model_settings.metadata:
            extra_kwargs["metadata"] = copy(model_settings.metadata)
        if model_settings.extra_body and isinstance(model_settings.extra_body, dict):
            extra_kwargs.update(model_settings.extra_body)

        # Add kwargs from model_settings.extra_args, filtering out None values
        if model_settings.extra_args:
            extra_kwargs.update(model_settings.extra_args)

        ret = await litellm.acompletion(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools or None,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            max_tokens=model_settings.max_tokens,
            tool_choice=self._remove_not_given(tool_choice),
            response_format=self._remove_not_given(response_format),
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            stream_options=stream_options,
            reasoning_effort=reasoning_effort,
            top_logprobs=model_settings.top_logprobs,
            extra_headers={**HEADERS, **(model_settings.extra_headers or {})},
            api_key=self.api_key,
            base_url=self.base_url,
            **extra_kwargs,
        )

        if isinstance(ret, litellm.types.utils.ModelResponse):
            return ret

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=cast("Literal['auto', 'required', 'none']", tool_choice)
            if tool_choice != NOT_GIVEN
            else "auto",
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret

    def _remove_not_given(self, value: Any) -> Any:
        if isinstance(value, NotGiven):
            return None
        return value


DEFAULT_MODEL_TYPE = LitellmModel


class OpenAIAgent(AnyAgent):
    """OpenAI agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: Agent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.OPENAI

    def _get_model(
        self,
        agent_config: AgentConfig,
    ) -> Model:
        """Get the model configuration for an OpenAI agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        return model_type(
            model=agent_config.model_id,
            base_url=agent_config.api_base,
            api_key=agent_config.api_key,
        )

    async def _load_agent(self) -> None:
        """Load the OpenAI agent with the given configuration."""
        if not agents_available:
            msg = "You need to `pip install 'any-agent[openai]'` to use this agent"
            raise ImportError(msg)
        if not agents_available:
            msg = "You need to `pip install openai-agents` to use this agent"
            raise ImportError(msg)

        tools = await self._load_tools(self.config.tools)

        kwargs_ = self.config.agent_args or {}
        if self.config.model_args:
            kwargs_["model_settings"] = ModelSettings(**self.config.model_args)
        if self.config.output_type:
            kwargs_["output_type"] = self.config.output_type

        self._tools = tools
        self._agent = Agent(
            name=self.config.name,
            instructions=self.config.instructions,
            model=self._get_model(self.config),
            tools=tools,
            mcp_servers=[],  # No longer needed with unified approach
            **kwargs_,
        )

    def _filter_mcp_tools(self, tools: list[Any], mcp_clients: list[Any]) -> list[Any]:
        """OpenAI framework doesn't expect the mcp tool to be included in `tools`."""
        # With the new MCPClient approach, MCP tools are already converted to regular callables
        # and included in the tools list, so we don't need to filter them out anymore.
        # The OpenAI framework can handle them as regular tools.
        return tools

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        if not kwargs.get("max_turns"):
            kwargs["max_turns"] = math.inf
        result = await Runner.run(self._agent, prompt, **kwargs)
        return result.final_output  # type: ignore[no-any-return]

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, we need to recreate it with the new output type
        # The OpenAI agents library requires output_type to be set during construction
        if self._agent:
            # Store current state
            current_tools = self._tools

            # Recreate the agent with the new output type
            kwargs_ = self.config.agent_args or {}
            if self.config.model_args:
                kwargs_["model_settings"] = ModelSettings(**self.config.model_args)
            if output_type:
                kwargs_["output_type"] = output_type

            self._agent = Agent(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self._get_model(self.config),
                tools=current_tools,
                mcp_servers=[],  # No longer needed with unified approach
                **kwargs_,
            )
