from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

from any_llm import acompletion, completion
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.completion import ChoiceDelta as Delta

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

logger = logging.getLogger(__name__)

try:
    from langchain_core.language_models.chat_models import (
        BaseChatModel,
        agenerate_from_stream,
        generate_from_stream,
    )
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        BaseMessageChunk,
        ChatMessage,
        ChatMessageChunk,
        FunctionMessage,
        FunctionMessageChunk,
        HumanMessage,
        HumanMessageChunk,
        SystemMessage,
        SystemMessageChunk,
        ToolCall,
        ToolCallChunk,
        ToolMessage,
    )
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        ChatResult,
    )
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.prebuilt import create_react_agent

    DEFAULT_AGENT_TYPE = create_react_agent

    langchain_available = True
except ImportError:
    langchain_available = False

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models import LanguageModelInput, LanguageModelLike
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from pydantic import BaseModel


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    if role == "assistant":
        content = _dict.get("content", "") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "system":
        return SystemMessage(content=_dict["content"])
    if role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    if role == "tool":
        return ToolMessage(content=_dict["content"], tool_call_id=_dict["tool_call_id"])
    return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    delta: Delta, default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = delta.role
    content = delta.content or ""
    additional_kwargs: dict[str, Any] = {}
    if delta.function_call:
        additional_kwargs["function_call"] = dict(delta.function_call)
    reasoning = getattr(delta, "reasoning", None)
    if reasoning and reasoning.content:
        additional_kwargs["reasoning_content"] = reasoning.content

    tool_call_chunks = []
    if raw_tool_calls := delta.tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                ToolCallChunk(
                    name=rtc.function.name if rtc.function else "",
                    args=rtc.function.arguments if rtc.function else "",
                    id=rtc.id,
                    index=rtc.index,
                )
                for rtc in raw_tool_calls
                if rtc.function
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if default_class == FunctionMessageChunk:
        if delta.function_call:
            return FunctionMessageChunk(
                content=delta.function_call.arguments or "",
                name=delta.function_call.name or "",
            )
    if role == "tool" or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    message_dict: dict[str, Any] = {"content": message.content}
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        error_message = f"Got unknown type {message}"
        raise ValueError(error_message)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatAnyLLM(BaseChatModel):
    """Chat model that uses the AnyLLM API."""

    model: str
    api_key: str | None
    api_base: str | None
    model_kwargs: Any

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        if stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = completion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return self._create_chat_result(response)

    def _create_chat_result(self, response: ChatCompletion) -> ChatResult:
        resp_dict = response.model_dump()

        generations = []
        token_usage = response.usage
        for res in resp_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage) and token_usage:
                message.response_metadata = {"model_name": self.model}
                message.usage_metadata = UsageMetadata(
                    input_tokens=token_usage.prompt_tokens,
                    output_tokens=token_usage.completion_tokens,
                    total_tokens=token_usage.prompt_tokens
                    + token_usage.completion_tokens,
                )
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": res.get("finish_reason")},
            )
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "model": self.model,
            **self.model_kwargs,
        }
        if stop is not None:
            if "stop" in params:
                error_message = "`stop` found in both the input and default params."
                raise ValueError(error_message)
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        result = completion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(result, Iterator):
            error_message = f"Expected Iterator, got {type(result)}"
            raise ValueError(error_message)
        for chunk_item in result:
            chunk_dict: dict[str, Any] = chunk_item.model_dump()
            if len(chunk_dict["choices"]) == 0:
                continue
            delta = chunk_dict["choices"][0]["delta"]
            message_chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = message_chunk.__class__
            cg_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager:
                content = message_chunk.content
                if isinstance(content, str):
                    run_manager.on_llm_new_token(content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        result = await acompletion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(result, AsyncIterator):
            error_message = f"Expected AsyncIterator, got {type(result)}"
            raise ValueError(error_message)
        async for stream_chunk in result:
            if not isinstance(stream_chunk, ChatCompletionChunk):
                error_message = "Unexpected chunk type"
                raise ValueError(error_message)
            for choice in stream_chunk.choices:
                delta = choice.delta
                message_chunk = _convert_delta_to_message_chunk(
                    delta, default_chunk_class
                )
                default_chunk_class = message_chunk.__class__
                cg_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    content = message_chunk.content
                    if isinstance(content, str):
                        await run_manager.on_llm_new_token(content, chunk=cg_chunk)
                yield cg_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion(messages=message_dicts, **params)  # type: ignore[arg-type]
        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool
        ],
        tool_choice: dict[str, Any] | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            **self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "anyllm-chat"


DEFAULT_MODEL_TYPE = ChatAnyLLM


class LangchainAgent(AnyAgent):
    """LangChain agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: CompiledStateGraph[Any] | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LANGCHAIN

    def _get_model(self, agent_config: AgentConfig) -> LanguageModelLike:
        """Get the model configuration for a LangChain agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        return cast(
            "LanguageModelLike",
            model_type(
                model=agent_config.model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                model_kwargs=model_args,
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LangChain agent with the given configuration."""
        if not langchain_available:
            msg = "You need to `pip install 'any-agent[langchain]'` to use this agent"
            raise ImportError(msg)

        imported_tools = await self._load_tools(self.config.tools)

        self._tools = imported_tools
        agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
        agent_args = self.config.agent_args or {}
        self._agent = agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=imported_tools,
            prompt=self.config.instructions,
            **agent_args,
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        inputs = {"messages": [("user", prompt)]}
        result = await self._agent.ainvoke(inputs, **kwargs)

        if not result.get("messages"):
            msg = "No messages returned from the agent."
            raise ValueError(msg)

        # Post-process for structured output if needed
        # This emulates the langgraph behavior for structured outputs,
        # but since it happens outside of langgraph we can control the model call,
        # because some providers like mistral don't allow for the model to be called without the most recent message being a user message.
        if self.config.output_type:
            # Add a follow-up message to request structured output
            structured_output_message = {
                "role": "user",
                "content": f"Please conform your output to the following schema: {self.config.output_type.model_json_schema()}.",
            }
            completion_params: dict[str, Any] = {}
            if self.config.model_args:
                # only include the temperature and frequency_penalty, not anything related to tools
                completion_params["temperature"] = self.config.model_args.get(
                    "temperature"
                )
                completion_params["frequency_penalty"] = self.config.model_args.get(
                    "frequency_penalty"
                )

            completion_params["model"] = self.config.model_id
            previous_messages = [
                _convert_message_to_dict(m) for m in result["messages"]
            ]
            completion_params["messages"] = [
                *previous_messages,
                structured_output_message,
            ]

            completion_params["response_format"] = self.config.output_type

            response = await self.call_model(**completion_params)
            return self.config.output_type.model_validate_json(
                response.choices[0].message.content or ""
            )
        return str(result["messages"][-1].content)

    async def call_model(self, **kwargs: Any) -> ChatCompletion:
        result = await acompletion(**kwargs)
        if not isinstance(result, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(result)}"
            raise ValueError(error_message)
        return result

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type
