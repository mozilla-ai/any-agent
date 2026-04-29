"""Thin shim around `mozilla-ai-tinyagent` to keep `any-agent`'s framework abstraction.

The TinyAgent loop now lives in the standalone `tinyagent` package
(PyPI: ``mozilla-ai-tinyagent``). This module re-exposes it through the
`AnyAgent` framework interface so existing `any-agent` callers keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tinyagent.agent import (
    DEFAULT_SYSTEM_PROMPT,  # noqa: F401  (re-exported for backwards compat)
    ToolExecutor,  # noqa: F401  (re-exported for backwards compat)
    final_answer,  # noqa: F401  (re-exported for backwards compat)
)
from tinyagent.agent import TinyAgent as _TinyAgentImpl

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

if TYPE_CHECKING:
    from pydantic import BaseModel


class TinyAgent(AnyAgent):
    """A lightweight agent implementation using `any-llm`.

    Modeled after the JS implementation https://huggingface.co/blog/tiny-agents.

    The agent loop is provided by the standalone `tinyagent` package; this class
    adapts it to the `AnyAgent` framework interface used by `any-agent`.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._inner = _TinyAgentImpl(config)

    @property
    def llm(self) -> Any:
        return self._inner.llm

    @property
    def uses_openai(self) -> bool:
        return self._inner.uses_openai

    @property
    def completion_params(self) -> dict[str, Any]:
        return self._inner.completion_params

    @property
    def clients(self) -> dict[str, ToolExecutor]:
        return self._inner.clients

    @clients.setter
    def clients(self, value: dict[str, ToolExecutor]) -> None:
        self._inner.clients = value

    async def _load_agent(self) -> None:
        await self._inner._load_agent()
        self._tools = self._inner._tools
        self._mcp_clients.extend(self._inner._mcp_clients)

    async def _run_async(
        self, prompt: str | list[dict[str, Any]], **kwargs: Any
    ) -> str | BaseModel:
        return await self._inner._run_async(prompt, **kwargs)

    async def call_model(self, **completion_params: Any) -> Any:
        return await self._inner.call_model(**completion_params)

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        await self._inner.update_output_type_async(output_type)

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.TINYAGENT
