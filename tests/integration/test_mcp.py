import os
from datetime import datetime
from typing import Any

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import MCPStdioParams


def get_current_year() -> str:
    """Get the current year"""
    return str(datetime.now().year)


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_mcp(agent_framework: AgentFramework, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Get the current year"""
    kwargs: dict[str, Any] = {}

    tools = [
        get_current_year,
        MCPStdioParams(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "--mount",
                f"type=bind,src={tmp_path},dst=/projects",
                "mcp/filesystem",
                "/projects",
            ],
            tools=[
                "write_file",
            ],
        ),
    ]
    agent_config = AgentConfig(
        model_id="gpt-4.1-mini",
        tools=tools,  # type: ignore[arg-type]
        **kwargs,
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    assert len(agent._mcp_servers) > 0
    result = agent.run(
        "Use the tools to find what year is it and write the value (single number) to /projects/tmp.txt"
    )
    assert os.path.exists(os.path.join(tmp_path, "tmp.txt"))
    with open(os.path.join(tmp_path, "tmp.txt")) as f:
        content = f.read()
    assert content == str(datetime.now().year)
    assert result
