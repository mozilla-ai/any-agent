# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
"""Wrapper for the TinyAgent framework.

The actual wrapping logic lives in the standalone `tinyagent` package; this
module adapts it so the wrapper operates on `agent._inner` (the underlying
`tinyagent.TinyAgent`) which is where the loop actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinyagent.callbacks.wrapper import _TinyAgentWrapper as _InnerWrapper

if TYPE_CHECKING:
    from any_agent.frameworks.tinyagent import TinyAgent


class _TinyAgentWrapper(_InnerWrapper):
    async def wrap(self, agent: TinyAgent) -> None:
        await super().wrap(agent._inner)

    async def unwrap(self, agent: TinyAgent) -> None:
        await super().unwrap(agent._inner)
