# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

from typing import Any


class Callback:
    """Base class for all callbacks."""

    def before_llm_call(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return context

    def before_tool_execution(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return context

    def after_llm_call(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return context

    def after_tool_execution(
        self, context: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return context
