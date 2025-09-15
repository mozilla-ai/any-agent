# mypy: disable-error-code="union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from any_llm import acompletion

from any_agent.config import AgentConfig, AgentFramework
from any_agent.tools.final_output import prepare_final_output

from .any_agent import AnyAgent

try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    adk_available = True
except ImportError:
    adk_available = False

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
    from google.adk.models.llm_request import LlmRequest
    from pydantic import BaseModel

ADK_TO_ANY_LLM_ROLE: dict[str, str] = {
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
}


def _safe_json_serialize(obj: Any) -> str:
    """Convert any Python object to a JSON-serializable type or string.

    Args:
    obj: The object to serialize.

    Returns:
        The JSON-serialized object string or string.

    """
    try:
        # Try direct JSON serialization first
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, OverflowError):
        return str(obj)


class AnyLlm(BaseLlm):
    """Wrapper around any-llm."""

    _kwargs: dict[str, Any]

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model)
        self._kwargs = kwargs or {}

    @staticmethod
    def _messages_from_content(llm_request: LlmRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for content in llm_request.contents:
            message: dict[str, Any] = {
                "role": ADK_TO_ANY_LLM_ROLE[str(content.role)],
            }
            message_content: list[Any] = []
            tool_calls: list[Any] = []
            if parts := content.parts:
                for part in parts:
                    if part.function_response:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.function_response.id,
                                "content": _safe_json_serialize(
                                    part.function_response.response
                                ),
                            }
                        )
                    elif part.text:
                        message_content.append({"type": "text", "text": part.text})
                    elif (
                        part.inline_data
                        and part.inline_data.data
                        and part.inline_data.mime_type
                    ):
                        # TODO Handle multimodal input
                        msg = f"Part of type {part.inline_data.mime_type} is not supported."
                        raise NotImplementedError(msg)
                    elif part.function_call:
                        tool_calls.append(
                            {
                                "type": "function",
                                "id": part.function_call.id,
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": _safe_json_serialize(
                                        part.function_call.args
                                    ),
                                },
                            }
                        )

            message["content"] = message_content or None
            message["tool_calls"] = tool_calls or None
            # messages from function_response were directly appended before
            if message["content"] or message["tool_calls"]:
                messages.append(message)

        return messages

    def _schema_to_dict(self, schema: types.Schema) -> dict[str, Any]:
        """Recursively converts a types.Schema to a pure-python dict.

        All enum values will be written as lower-case strings.

        Args:
            schema: The schema to convert.

        Returns:
            The dictionary representation of the schema.

        """
        # Dump without json encoding so we still get Enum members
        schema_dict = schema.model_dump(exclude_none=True)

        # ---- normalise this level ------------------------------------------------
        if "type" in schema_dict:
            # schema_dict["type"] can be an Enum or a str
            t = schema_dict["type"]
            schema_dict["type"] = (t.value if isinstance(t, types.Type) else t).lower()

        # ---- recurse into `items` -----------------------------------------------
        if "items" in schema_dict:
            schema_dict["items"] = self._schema_to_dict(
                schema.items
                if isinstance(schema.items, types.Schema)
                else types.Schema.model_validate(schema_dict["items"])
            )

        # ---- recurse into `properties` ------------------------------------------
        if "properties" in schema_dict:
            new_props = {}
            for key, value in schema_dict["properties"].items():
                # value is a dict → rebuild a Schema object and recurse
                if isinstance(value, dict):
                    new_props[key] = self._schema_to_dict(
                        types.Schema.model_validate(value)
                    )
                # value is already a Schema instance
                elif isinstance(value, types.Schema):
                    new_props[key] = self._schema_to_dict(value)
                # plain dict without nested schemas
                else:
                    new_props[key] = value
                    if "type" in new_props[key]:
                        new_props[key]["type"] = new_props[key]["type"].lower()
            schema_dict["properties"] = new_props

        return schema_dict

    def _function_declaration_to_tool_param(
        self,
        function_declaration: types.FunctionDeclaration,
    ) -> dict[str, Any]:
        """Convert a types.FunctionDeclaration to a openapi spec dictionary.

        Args:
            function_declaration: The function declaration to convert.

        Returns:
            The openapi spec dictionary representation of the function declaration.

        """
        assert function_declaration.name

        properties = {}
        if (
            function_declaration.parameters
            and function_declaration.parameters.properties
        ):
            for key, value in function_declaration.parameters.properties.items():
                properties[key] = self._schema_to_dict(value)

        return {
            "type": "function",
            "function": {
                "name": function_declaration.name,
                "description": function_declaration.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }

    def _llm_request_to_completion_args(
        self, llm_request: LlmRequest
    ) -> dict[str, Any]:
        messages = self._messages_from_content(llm_request)
        if llm_request.config.system_instruction:
            messages.insert(
                0, {"role": "system", "content": llm_request.config.system_instruction}
            )

        completion_args: dict[str, Any] = {"messages": messages, "model": self.model}

        if config := llm_request.config:
            if config.tools and config.tools[0].function_declarations:
                completion_args["tools"] = [
                    self._function_declaration_to_tool_param(tool)
                    for tool in config.tools[0].function_declarations
                ]

            for name in (
                "frequency_penalty",
                "presence_penalty",
                "response_formattemperature",
                "top_k",
                "top_p",
            ):
                if attr := getattr(config, name, None):
                    completion_args[name] = attr

            if max_tokens := config.max_output_tokens:
                completion_args["max_tokens"] = max_tokens
            if stop := config.stop_sequences:
                completion_args["stop"] = stop

        return {**self._kwargs, **completion_args}

    def _completion_to_llm_response(self, completion: ChatCompletion) -> LlmResponse:
        llm_response = self._message_to_response(completion.choices[0].message)
        if usage := completion.usage:
            llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.prompt_tokens,
                candidates_token_count=usage.completion_tokens,
                total_token_count=usage.total_tokens,
            )
        return llm_response

    def _message_to_response(
        self, message: ChatCompletionMessage, is_partial: bool = False
    ) -> LlmResponse:
        parts = []
        if content := message.content:
            parts.append(types.Part.from_text(text=content))

        if tool_calls := message.tool_calls:
            for tool_call in tool_calls:
                if tool_call.type == "function":
                    part = types.Part.from_function_call(
                        name=tool_call.function.name,
                        args=json.loads(tool_call.function.arguments or "{}"),
                    )
                    part.function_call.id = tool_call.id
                    parts.append(part)

        return LlmResponse(
            content=types.Content(role="model", parts=parts), partial=is_partial
        )

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate one content from the given contents and tools.

        Args:
          llm_request: LlmRequest, the request to send to the LLM.
          stream: bool = False, whether to do streaming call.

        Yields:
          a generator of types.Content.

          For non-streaming call, it will only yield one Content.

          For streaming call, it may yield more than one content, but all yielded
          contents should be treated as one content by merging the
          parts list.

        """
        completion_args = self._llm_request_to_completion_args(llm_request)
        if stream:
            pass
        else:
            completion = await acompletion(**completion_args)
            yield self._completion_to_llm_response(completion)  # type: ignore[arg-type]


DEFAULT_MODEL_TYPE = AnyLlm


class GoogleAgent(AnyAgent):
    """Google ADK agent implementation that handles both loading and running."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._agent: LlmAgent | None = None

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.GOOGLE

    def _get_model(self, agent_config: AgentConfig) -> BaseLlm:
        """Get the model configuration for a Google agent."""
        model_type = agent_config.model_type or DEFAULT_MODEL_TYPE
        model_args = agent_config.model_args or {}
        if self.config.output_type:
            model_args["tool_choice"] = "required"
        return model_type(
            model=agent_config.model_id,
            api_key=agent_config.api_key,
            api_base=agent_config.api_base,
            **model_args,
        )

    async def _load_agent(self) -> None:
        """Load the Google agent with the given configuration."""
        if not adk_available:
            msg = "You need to `pip install 'any-agent[google]'` to use this agent"
            raise ImportError(msg)

        tools = await self._load_tools(self.config.tools)

        agent_type = self.config.agent_type or LlmAgent

        self._tools = tools

        instructions = self.config.instructions or ""
        if self.config.output_type:
            instructions, final_output_tool = prepare_final_output(
                self.config.output_type, instructions
            )
            tools.append(final_output_tool)

        self._agent = agent_type(
            name=self.config.name,
            instruction=instructions,
            model=self._get_model(self.config),
            tools=tools,
            **self.config.agent_args or {},
            output_key="response",
        )

    async def _run_async(  # type: ignore[no-untyped-def]
        self,
        prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> str | BaseModel:
        if not self._agent:
            error_message = "Agent not loaded. Call load_agent() first."
            raise ValueError(error_message)
        runner = InMemoryRunner(self._agent)
        user_id = user_id or str(uuid4())
        session_id = session_id or str(uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if self.config.output_type:
            final_output = None
            final_output_attempts = 0
            # We allow for two retries: one to make it a proper json string, and one to make it a valid pydantic model
            max_output_attempts = 3

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                **kwargs,
            ):
                if not event.content or not event.content.parts:
                    continue

                # Check for final_output function responses
                for part in event.content.parts:
                    if (
                        part.function_response
                        and part.function_response.name == "final_output"
                        and part.function_response.response
                    ):
                        final_output_attempts += 1
                        if part.function_response.response.get("success"):
                            final_output = part.function_response.response.get("result")
                            break
                        if final_output_attempts >= max_output_attempts:
                            msg = f"Final output failed after {final_output_attempts} attempts"
                            raise ValueError(msg)

                if final_output or final_output_attempts >= max_output_attempts:
                    break

            if not final_output:
                msg = "No final response found"
                raise ValueError(msg)
            return self.config.output_type.model_validate(final_output)

        async for _ in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            **kwargs,
        ):
            pass
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        assert session, "Session should not be None"
        return str(session.state.get("response"))

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

        # If agent is already loaded, we need to recreate it with the new output type
        # The Google ADK agent requires output_type to be set during construction
        if self._agent:
            # Store current tools (excluding any existing final_output tool)
            current_tools = [
                tool
                for tool in self._tools
                if not (hasattr(tool, "__name__") and tool.__name__ == "final_output")
            ]

            # Prepare instructions and tools for the new output type
            instructions = self.config.instructions or ""
            if output_type:
                instructions, final_output_tool = prepare_final_output(
                    output_type, instructions
                )
                current_tools.append(final_output_tool)

            # Recreate the agent with the new configuration
            agent_type = self.config.agent_type or LlmAgent
            self._agent = agent_type(
                name=self.config.name,
                instruction=instructions,
                model=self._get_model(self.config),
                tools=current_tools,
                **self.config.agent_args or {},
                output_key="response",
            )

            # Update the tools list
            self._tools = current_tools
