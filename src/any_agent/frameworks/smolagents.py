# mypy: disable-error-code="union-attr"
import json
from collections.abc import Callable, Generator
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from any_agent.config import AgentConfig, AgentFramework
from any_agent.frameworks.any_agent import AnyAgent
from any_agent.tools.final_output import prepare_final_output

try:
    from smolagents import (
        FinalAnswerTool,
        Tool,
        ToolCallingAgent,
    )
    from smolagents.models import (
        ApiModel,
        ChatMessage,
        ChatMessageStreamDelta,
        ChatMessageToolCallStreamDelta,
    )
    from smolagents.monitoring import TokenUsage

    smolagents_available = True
except ImportError:
    smolagents_available = False

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
    from smolagents import MultiStepAgent


DEFAULT_AGENT_TYPE = ToolCallingAgent


class AnyLLMModel(ApiModel):
    """Smolagents ApiModel that delegates all requests to the any-llm backend.

    This class purposefully mirrors the methods API of
    `LiteLLMModel` so that it can be injected anywhere a Smolagents model is
    expected.
    Refer: https://github.com/huggingface/smolagents/blob/main/src/smolagents/models.py
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

        self._anyllm_common_kwargs: dict[str, Any] = {
            "model": model_id,
            "api_key": api_key,
            "api_base": api_base,
            **kwargs,
        }

    def create_client(self) -> Any:
        """Create the any-llm client, required method for ApiModel subclasses."""
        import any_llm

        return any_llm

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extend _prepare_completion_kwargs, ensuring messages are correctly serialized.

        For example, smolagents returns messages containing enum <MessageRole.ASSISTANT>
        which is transformed to string "assistant".
        """
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            tool_choice=tool_choice,
            **kwargs,
        )
        if "messages" in completion_kwargs:
            messages_json = json.dumps(
                completion_kwargs["messages"], default=pydantic_encoder
            )
            completion_kwargs["messages"] = json.loads(messages_json)
            for message in completion_kwargs["messages"]:
                # Convert content to plain string from the list of dicts for any-llm processing
                content = deepcopy(message.get("content", []))
                message["content"] = ""
                for item in content:
                    if item.get("type") == "text":
                        message["content"] += item["text"] + "\n"
        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        self._apply_rate_limit()  # type: ignore[no-untyped-call]
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,  # type: ignore[arg-type]
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        payload = {**self._anyllm_common_kwargs, **completion_kwargs}
        response: ChatCompletion = self.client.completion(**payload)

        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            ),
            raw=response.model_dump(),
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageStreamDelta]:
        self._apply_rate_limit()  # type: ignore[no-untyped-call]
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,  # type: ignore[arg-type]
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Ensure usage information is included in the streamed chunks when the provider supports it
        completion_kwargs.setdefault("stream_options", {"include_usage": True})

        payload = {**self._anyllm_common_kwargs, **completion_kwargs, "stream": True}
        response_iterator: Generator[ChatCompletionChunk] = self.client.completion(
            **payload
        )

        for event in response_iterator:
            if getattr(event, "usage", None):
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )

            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    tool_calls = None
                    if choice.delta.tool_calls:
                        tool_calls = [
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,  # type: ignore[arg-type]
                            )
                            for delta in choice.delta.tool_calls
                        ]

                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=tool_calls,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        error_message = f"No content or tool calls in event: {event}"
                        raise ValueError(error_message)


DEFAULT_MODEL_TYPE = AnyLLMModel


class SmolagentsAgent(AnyAgent):
    """Smolagents agent implementation that handles both loading and running."""

    def __init__(
        self,
        config: AgentConfig,
    ):
        super().__init__(config)
        self._agent: MultiStepAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.SMOLAGENTS

    def _get_model(self, agent_config: AgentConfig) -> Any:
        """Get the model configuration for a smolagents agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        kwargs = {
            "model_id": agent_config.model_id,
            "api_key": agent_config.api_key,
            "api_base": agent_config.api_base,
            **model_args,
        }
        return model_type(**kwargs)

    def _setup_output_type(self, output_type: type[BaseModel] | None) -> None:
        """Set up the output type handling for the agent.

        Args:
            output_type: The output type to set up, or None to remove output type constraint

        """
        if not self._agent:
            return

        if self.config.instructions:
            self._agent.prompt_templates["system_prompt"] = self.config.instructions

        if output_type:
            instructions, final_output_function = prepare_final_output(
                output_type, self.config.instructions
            )

            class FinalAnswerToolWrapper(FinalAnswerTool):  # type: ignore[no-untyped-call]
                def __init__(
                    self,
                    final_output_func: Callable[
                        [str], dict[str, str | bool | dict[str, Any] | list[Any]]
                    ],
                ):
                    super().__init__()  # type: ignore[no-untyped-call]
                    self.final_output_func = final_output_func
                    # Copying the __doc__ relies upon the final_output_func having a single str parameter called "answer"
                    if (
                        not self.final_output_func.__code__.co_varnames[0] == "answer"
                        or not self.final_output_func.__doc__
                    ):
                        msg = "The final_output_func must have a single parameter of type str"
                        raise ValueError(msg)

                    self.inputs = {
                        "answer": {
                            "type": "string",
                            "description": self.final_output_func.__doc__,
                        }
                    }

                def forward(self, answer: str) -> Any:
                    result = self.final_output_func(answer)
                    if result.get("success"):
                        return answer
                    raise ValueError(result["result"])

            self._agent.tools["final_answer"] = FinalAnswerToolWrapper(
                final_output_function
            )

            # Update the system prompt with the modified instructions
            if instructions:
                self._agent.prompt_templates["system_prompt"] = instructions

    async def _load_agent(self) -> None:
        """Load the Smolagents agent with the given configuration."""
        if not smolagents_available:
            msg = "You need to `pip install 'any-agent[smolagents]'` to use this agent"
            raise ImportError(msg)

        tools = await self._load_tools(self.config.tools)

        main_agent_type = self.config.agent_type or DEFAULT_AGENT_TYPE

        agent_args = self.config.agent_args or {}

        self._tools = tools
        self._agent = main_agent_type(
            name=self.config.name,
            model=self._get_model(self.config),
            tools=tools,
            verbosity_level=-1,  # OFF
            **agent_args,
        )

        if self.config.instructions:
            self._agent.prompt_templates["system_prompt"] = self.config.instructions

        # Set up output type handling
        self._setup_output_type(self.config.output_type)

        assert self._agent

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        result = self._agent.run(prompt, **kwargs)
        if self.config.output_type:
            result_json = (
                result.model_dump_json()
                if hasattr(result, "model_dump_json")
                else str(result)
            )
            return self.config.output_type.model_validate_json(result_json)
        return str(result)

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, update its output handling
        self._setup_output_type(output_type)
