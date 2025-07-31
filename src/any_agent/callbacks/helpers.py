# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

import json
from typing import Any

from opentelemetry.trace import Status, StatusCode


def serialize_for_attribute(data: Any) -> str:
    """Serialize data for OpenTelemetry attributes, handling various types safely."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)


def determine_output_type(output: Any) -> str:
    """Determine output type based on the output content."""
    if isinstance(output, str):
        try:
            json.loads(output)
        except json.JSONDecodeError:
            return "text"
    return "json"


def determine_tool_status(tool_output: str, output_type: str) -> Status | StatusCode:
    """Determine the status based on tool output content and type."""
    if output_type == "text" and "Error calling tool:" in tool_output:
        return Status(StatusCode.ERROR, description=tool_output)
    return StatusCode.OK
