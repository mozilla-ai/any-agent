import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from any_agent import (
    AgentConfig,
    AgentFramework,
    AnyAgent,
)
from any_agent.config import MCPStreamableHttp
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    group_spans,
)


class Step(BaseModel):
    number: int
    description: str


class Steps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]


def test_load_and_run_agent_streamable_http(
    agent_framework: AgentFramework,
    date_streamable_http_server: dict[str, Any],
    tmp_path: Path,
) -> None:
    kwargs = {}

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID

    tmp_file = "tmp.txt"

    def write_file(text: str) -> None:
        """write the text to a file in the tmp_path directory

        Args:
            text (str): The text to write to the file.

        Returns:
            None
        """
        (tmp_path / tmp_file).write_text(text)

    tools = [
        write_file,
        MCPStreamableHttp(
            url=date_streamable_http_server["url"],
            client_session_timeout_seconds=30,
        ),
    ]
    agent_config = AgentConfig(
        tools=tools,  # type: ignore[arg-type]
        instructions="Use the available tools to answer.",
        model_args=get_default_agent_model_args(agent_framework),
        output_type=Steps,
        **kwargs,  # type: ignore[arg-type]
    )
    agent = AnyAgent.create(agent_framework, agent_config)

    agent_trace = agent.run(
        "First, find what year it is in the America/New_York timezone. "
        "Then, write the value (single number) to a file. "
        "Finally, return a list of the steps you have taken.",
    )

    assert isinstance(agent_trace.final_output, Steps)

    assert (tmp_path / tmp_file).read_text() == str(datetime.now().year)
    _, _, tool_executions = group_spans(agent_trace.spans)

    assert len(tool_executions) >= 1
    tool_args_raw = tool_executions[0].attributes.get("gen_ai.tool.args")
    assert tool_args_raw is not None
    args = json.loads(tool_args_raw)
    assert "timezone" in args
