from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span


def _set_tool_output(tool_output: Any, span: Span) -> None:
    if tool_output is None:
        tool_output = "{}"

    if isinstance(tool_output, str):
        try:
            json.loads(tool_output)
            output_type = "json"
        except json.decoder.JSONDecodeError:
            output_type = "text"
    else:
        try:
            tool_output = json.dumps(tool_output, default=str, ensure_ascii=False)
            output_type = "json"
        except TypeError:
            tool_output = str(tool_output)
            output_type = "text"

    span.set_attributes(
        {"gen_ai.output": tool_output, "gen_ai.output.type": output_type}
    )
