import json
import uuid
from collections.abc import Sequence
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, cast

from any_llm import AnyLLM, acompletion
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from any_agent import AgentConfig, AgentFramework
from any_agent.logging import logger
from any_agent.tools.final_output import prepare_final_output
from any_agent.vendor.llama_index_utils import (
    force_single_tool_call,
    from_openai_message_dict,
    to_openai_message_dicts,
    update_tool_calls,
)

from .any_agent import AnyAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.llms import LLM

try:
    from llama_index.core.agent.workflow import (
        BaseWorkflowAgent,
        FunctionAgent,
        ReActAgent,
    )
    from llama_index.core.base.llms.generic_utils import (
        achat_to_completion_decorator,
        astream_chat_to_completion_decorator,
        chat_to_completion_decorator,
        stream_chat_to_completion_decorator,
    )
    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        ChatResponseAsyncGen,
        ChatResponseGen,
        CompletionResponse,
        CompletionResponseAsyncGen,
        CompletionResponseGen,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.constants import DEFAULT_TEMPERATURE
    from llama_index.core.llms.callbacks import (
        llm_chat_callback,
        llm_completion_callback,
    )
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.tools import ToolSelection
    from llama_index.core.tools.types import BaseTool

    DEFAULT_AGENT_TYPE = FunctionAgent

    class AnyLLMWrapper(FunctionCallingLLM):
        """LlamaIndex LLM wrapper that uses any_llm. Adapted from the litellm implementation in the llama_index.llms.litellm module."""

        model: str = Field(
            description="Model identifier in provider:model format (e.g., 'openai:gpt-4')"
        )
        temperature: float = Field(
            default=DEFAULT_TEMPERATURE,
            description="The temperature to use during generation.",
            ge=0.0,
            le=2.0,
        )
        max_tokens: int | None = Field(
            default=None,
            description="The maximum number of tokens to generate.",
            gt=0,
        )
        additional_kwargs: dict[str, Any] = Field(
            default_factory=dict,
            description="Additional kwargs for the LLM API.",
        )
        max_retries: int = Field(
            default=10, description="The maximum number of API retries."
        )

        _client: AnyLLM | None = PrivateAttr(default=None)
        _provider: str = PrivateAttr(default="")
        _model_name: str = PrivateAttr(default="")

        def __init__(
            self,
            model: str,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int | None = None,
            additional_kwargs: dict[str, Any] | None = None,
            max_retries: int = 10,
            api_key: str | None = None,
            api_base: str | None = None,
            **kwargs: Any,
        ) -> None:
            additional_kwargs = additional_kwargs or {}

            super().__init__(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_kwargs=additional_kwargs,
                max_retries=max_retries,
                **kwargs,
            )

            self._parse_model(model)
            self._client = AnyLLM.create(
                provider=self._provider,
                api_key=api_key,
                api_base=api_base,
            )

        def _parse_model(self, model: str) -> None:
            provider, model_name = AnyLLM.split_model_provider(model)

            self._provider = provider
            self._model_name = model_name

        @classmethod
        def class_name(cls) -> str:
            return "any_llm_wrapper"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,  # if it's not, it can't be an agent
                model_name=self.model,
            )

        def _prepare_chat_with_tools(
            self,
            tools: Sequence[BaseTool],
            user_msg: str | ChatMessage | None = None,
            chat_history: list[ChatMessage] | None = None,
            verbose: bool = False,
            allow_parallel_tool_calls: bool = False,
            tool_required: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            tool_specs = [
                tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
            ]

            if isinstance(user_msg, str):
                user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

            messages = chat_history or []
            if user_msg:
                messages.append(user_msg)
            return {
                "messages": messages,
                "tools": tool_specs or None,
                "parallel_tool_calls": allow_parallel_tool_calls,
                "tool_choice": "required" if tool_required else "auto",
                **kwargs,
            }

        def _validate_chat_with_tools_response(
            self,
            response: ChatResponse,
            tools: Sequence[BaseTool],
            allow_parallel_tool_calls: bool = False,
            **kwargs: Any,
        ) -> ChatResponse:
            if not allow_parallel_tool_calls:
                force_single_tool_call(response)
            return response

        def get_tool_calls_from_response(
            self,
            response: ChatResponse,
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
        ) -> list[ToolSelection]:
            tool_calls = response.message.additional_kwargs.get("tool_calls", [])
            if len(tool_calls) < 1:
                if error_on_no_tool_call:
                    msg = f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                    raise ValueError(msg)
                return []

            tool_selections = []
            for tool_call in tool_calls:
                if tool_call["type"] != "function" or "function" not in tool_call:
                    msg = f"Invalid tool call of type {tool_call['type']}"
                    raise ValueError(msg)

                function = tool_call.get("function", {})
                tool_name = function.get("name")
                arguments = function.get("arguments")

                try:
                    if arguments:
                        argument_dict = json.loads(arguments)
                    else:
                        argument_dict = {}
                except (ValueError, TypeError, JSONDecodeError):
                    argument_dict = {}

                if tool_name:
                    tool_selections.append(
                        ToolSelection(
                            tool_id=tool_call.get("id") or str(uuid.uuid4()),
                            tool_name=tool_name,
                            tool_kwargs=argument_dict,
                        )
                    )
            if len(tool_selections) == 0 and error_on_no_tool_call:
                msg = "No valid tool calls found."
                raise ValueError(msg)

            return tool_selections

        @property
        def _model_kwargs(self) -> dict[str, Any]:
            base_kwargs = {
                "model": self._model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            return {
                **base_kwargs,
                **self.additional_kwargs,
            }

        def _get_all_kwargs(self, **kwargs: Any) -> dict[str, Any]:
            return {
                **self._model_kwargs,
                **kwargs,
            }

        @llm_chat_callback()
        def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
            return self._chat(messages, **kwargs)

        @llm_chat_callback()
        def stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatResponseGen:
            return self._stream_chat(messages, **kwargs)

        @llm_completion_callback()
        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            complete_fn = chat_to_completion_decorator(self._chat)
            return complete_fn(prompt, **kwargs)

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            stream_complete_fn = stream_chat_to_completion_decorator(self._stream_chat)
            return stream_complete_fn(prompt, **kwargs)

        def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
            if not self._client:
                msg = "Client not initialized."
                raise ValueError(msg)

            message_dicts = to_openai_message_dicts(messages)
            all_kwargs = self._get_all_kwargs(**kwargs)

            response_result = self._client.completion(
                messages=cast(
                    "list[dict[str, Any] | ChatCompletionMessage]", message_dicts
                ),
                stream=False,
                **all_kwargs,
            )

            response = cast("ChatCompletion", response_result)

            message_dict = response.choices[0].message.model_dump()
            message = from_openai_message_dict(message_dict)

            return ChatResponse(
                message=message,
                raw=response,
                additional_kwargs=self._get_response_token_counts(response),
            )

        def _stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatResponseGen:
            if not self._client:
                msg = "Client not initialized."
                raise ValueError(msg)

            message_dicts = to_openai_message_dicts(messages)
            all_kwargs = self._get_all_kwargs(**kwargs)
            if "max_tokens" in all_kwargs and all_kwargs["max_tokens"] is None:
                all_kwargs.pop("max_tokens")

            stream_result = self._client.completion(
                messages=cast(
                    "list[dict[str, Any] | ChatCompletionMessage]", message_dicts
                ),
                stream=True,
                **all_kwargs,
            )

            stream = cast("Iterator[ChatCompletionChunk]", stream_result)

            def gen() -> ChatResponseGen:
                content = ""
                tool_calls: list[dict[str, Any]] = []
                for response in stream:
                    if not response.choices or len(response.choices) == 0:
                        continue

                    delta = response.choices[0].delta
                    role = delta.role or MessageRole.ASSISTANT
                    content_delta = delta.content or ""
                    content += content_delta

                    tool_call_delta = None
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_call_delta = delta.tool_calls

                    if tool_call_delta is not None and len(tool_call_delta) > 0:
                        tool_calls = update_tool_calls(tool_calls, tool_call_delta)

                    additional_kwargs = {}
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=content_delta,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

            return gen()

        @llm_chat_callback()
        async def achat(
            self,
            messages: Sequence[ChatMessage],
            **kwargs: Any,
        ) -> ChatResponse:
            return await self._achat(messages, **kwargs)

        @llm_chat_callback()
        async def astream_chat(
            self,
            messages: Sequence[ChatMessage],
            **kwargs: Any,
        ) -> ChatResponseAsyncGen:
            return await self._astream_chat(messages, **kwargs)

        @llm_completion_callback()
        async def acomplete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            acomplete_fn = achat_to_completion_decorator(self._achat)
            return await acomplete_fn(prompt, **kwargs)

        @llm_completion_callback()
        async def astream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseAsyncGen:
            astream_complete_fn = astream_chat_to_completion_decorator(
                self._astream_chat
            )
            return await astream_complete_fn(prompt, **kwargs)

        async def _achat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatResponse:
            if not self._client:
                msg = "Client not initialized."
                raise ValueError(msg)

            message_dicts = to_openai_message_dicts(messages)
            all_kwargs = self._get_all_kwargs(**kwargs)
            if "max_tokens" in all_kwargs and all_kwargs["max_tokens"] is None:
                all_kwargs.pop("max_tokens")

            response_result = await self._client.acompletion(
                messages=cast(
                    "list[dict[str, Any] | ChatCompletionMessage]", message_dicts
                ),
                stream=False,
                **all_kwargs,
            )

            response = cast("ChatCompletion", response_result)

            message_dict = response.choices[0].message.model_dump()
            message = from_openai_message_dict(message_dict)

            return ChatResponse(
                message=message,
                raw=response,
                additional_kwargs=self._get_response_token_counts(response),
            )

        async def _astream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> ChatResponseAsyncGen:
            if not self._client:
                msg = "Client not initialized."
                raise ValueError(msg)

            message_dicts = to_openai_message_dicts(messages)
            all_kwargs = self._get_all_kwargs(**kwargs)
            if "max_tokens" in all_kwargs and all_kwargs["max_tokens"] is None:
                all_kwargs.pop("max_tokens")

            stream_result = await self._client.acompletion(
                messages=cast(
                    "list[dict[str, Any] | ChatCompletionMessage]", message_dicts
                ),
                stream=True,
                **all_kwargs,
            )

            stream = cast("AsyncIterator[ChatCompletionChunk]", stream_result)

            async def gen() -> ChatResponseAsyncGen:
                content = ""
                tool_calls: list[dict[str, Any]] = []
                async for response in stream:
                    if not response.choices or len(response.choices) == 0:
                        continue

                    delta = response.choices[0].delta
                    role = delta.role or MessageRole.ASSISTANT
                    content_delta = delta.content or ""
                    content += content_delta

                    tool_call_delta = None
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_call_delta = delta.tool_calls

                    if tool_call_delta is not None and len(tool_call_delta) > 0:
                        tool_calls = update_tool_calls(tool_calls, tool_call_delta)

                    additional_kwargs = {}
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=content_delta,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

            return gen()

        def _get_response_token_counts(
            self, raw_response: ChatCompletion | ChatCompletionChunk
        ) -> dict[str, Any]:
            if not raw_response.usage:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            return raw_response.usage.model_dump()

        @property
        def _is_chat_model(self) -> bool:
            return True

    DEFAULT_MODEL_TYPE = AnyLLMWrapper
    llama_index_available = True
except ImportError:
    llama_index_available = False


class LlamaIndexAgent(AnyAgent):
    """LLamaIndex agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: BaseWorkflowAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.LLAMA_INDEX

    def _get_model(self, agent_config: AgentConfig) -> "LLM":
        """Get the model configuration for a llama_index agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        additional_kwargs = agent_config.model_args or {}

        model_id = agent_config.model_id

        return cast(
            "LLM",
            model_type(
                model=model_id,
                api_key=agent_config.api_key,
                api_base=agent_config.api_base,
                additional_kwargs=additional_kwargs,  # type: ignore[arg-type]
            ),
        )

    async def _load_agent(self) -> None:
        """Load the LLamaIndex agent with the given configuration."""
        if not llama_index_available:
            msg = "You need to `pip install 'any-agent[llama_index]'` to use this agent"
            raise ImportError(msg)

        instructions = self.config.instructions
        tools_to_use = list(self.config.tools)
        if self.config.output_type:
            instructions, final_output_function = prepare_final_output(
                self.config.output_type, instructions
            )
            tools_to_use.append(final_output_function)
        imported_tools = await self._load_tools(tools_to_use)
        agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
        # if agent type is FunctionAgent but there are no tools, throw an error
        if agent_type == FunctionAgent and not imported_tools:
            logger.warning(
                "FunctionAgent requires tools and none were provided. Using ReActAgent instead."
            )
            agent_type = ReActAgent
        self._tools = imported_tools

        self._agent = agent_type(
            name=self.config.name,
            tools=imported_tools,
            description=self.config.description or "The main agent",
            llm=self._get_model(self.config),
            system_prompt=instructions,
            **self.config.agent_args or {},
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result: AgentOutput = await self._agent.run(prompt, **kwargs)
        # assert that it's a TextBlock
        if not result.response.blocks or not hasattr(result.response.blocks[0], "text"):
            msg = f"Agent did not return a valid response: {result.response}"
            raise ValueError(msg)
        if self.config.output_type:
            # First try to validate the output directly
            try:
                return self.config.output_type.model_validate_json(
                    result.response.blocks[0].text
                )
            except ValidationError:
                # If validation fails, call the model again to enforce structured output
                completion_params = self.config.model_args or {}
                completion_params["model"] = self.config.model_id
                model_output_message = {
                    "role": "assistant",
                    "content": result.response.blocks[0].text,
                }
                structured_output_message = {
                    "role": "user",
                    "content": f"Please conform your output to the following schema: {self.config.output_type.model_json_schema()}.",
                }
                completion_params["messages"] = [
                    model_output_message,
                    structured_output_message,
                ]
                completion_params["response_format"] = self.config.output_type
                response = await self.call_model(**completion_params)
                return self.config.output_type.model_validate_json(
                    response.choices[0].message.content
                )
        return result.response.blocks[0].text

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, we need to recreate it with the new output type
        # The LlamaIndex agent requires output_type tools to be set during construction
        if self._agent:
            # Prepare instructions and tools for the new output type
            instructions = self.config.instructions
            tools_to_use = list(self.config.tools)
            if output_type:
                instructions, final_output_function = prepare_final_output(
                    output_type, instructions
                )
                tools_to_use.append(final_output_function)

            # Recreate the agent with the new configuration
            agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE
            # if agent type is FunctionAgent but there are no tools, throw an error
            if agent_type == FunctionAgent and not tools_to_use:
                logger.warning(
                    "FunctionAgent requires tools and none were provided. Using ReActAgent instead."
                )
                agent_type = ReActAgent

            # We need to reload the tools since they might have changed
            imported_tools = await self._load_tools(tools_to_use)
            self._tools = imported_tools

            self._agent = agent_type(
                name=self.config.name,
                tools=imported_tools,
                description=self.config.description or "The main agent",
                llm=self._get_model(self.config),
                system_prompt=instructions,
                **self.config.agent_args or {},
            )

    async def call_model(self, **kwargs: Any) -> Any:
        return await acompletion(**kwargs)
