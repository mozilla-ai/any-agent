import os
import tempfile as tmpfile
import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent

frameworks = [item for item in AgentFramework]


@pytest.mark.parametrize(
    "framework",
    frameworks,
)
@pytest.mark.skipif(
    os.environ.get("MCP_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `MCP_INTEGRATION_TESTS=TRUE` env var",
)
def test_mcp(framework):
    agent_framework = AgentFramework(framework)
    kwargs = {}

    tmp_dir = tmpfile.mkdtemp()
    tools = [
        {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "--mount",
                f"type=bind,src={tmp_dir},dst=/projects",
                "mcp/filesystem",
                "/projects",
            ],
            "tools": [
                "write_file",
            ],
        },
        {"command": "docker", "args": ["run", "-i", "--rm", "mcp/time"]},
    ]
    agent_config = AgentConfig(
        model_id="gpt-4o",
        tools=tools,
        **kwargs,
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    assert len(agent.tools) > 0
    result = agent.run("Write 'hi' to /projects/tmp.txt")
    # Check if the file was created
    assert os.path.exists(os.path.join(tmp_dir, "tmp.txt"))
    # Check if the content is correct
    with open(os.path.join(tmp_dir, "tmp.txt"), "r") as f:
        content = f.read()
    assert content == "hi"
    assert result
    # remove the temporary directory
    os.remove(os.path.join(tmp_dir, "tmp.txt"))
