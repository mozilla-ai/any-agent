import os
import tempfile as tmpfile
from datetime import datetime
from typing import Any

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import MCPStdioParams
from any_agent.tools import search_web


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_mcp(agent_framework: AgentFramework) -> None:
    kwargs: dict[str, Any] = {}

    tmp_dir = tmpfile.mkdtemp()
    tools = [
        search_web,
        MCPStdioParams(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "--mount",
                f"type=bind,src={tmp_dir},dst=/projects",
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
        "Search the web to find 'what year is today' and write the value (single number) to /projects/tmp.txt"
    )
    # Check if the file was created
    assert os.path.exists(os.path.join(tmp_dir, "tmp.txt"))
    # Check if the content is correct
    with open(os.path.join(tmp_dir, "tmp.txt")) as f:
        content = f.read()
    assert content == str(datetime.now().year)
    assert result
    # remove the temporary directory
    os.remove(os.path.join(tmp_dir, "tmp.txt"))
