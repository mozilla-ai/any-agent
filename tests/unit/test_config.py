from any_agent.config import MCPTool, AgentConfig

# filepath: src/any_agent/tests/test_config.py


def test_mcp_tool_without_tools():
    """Test that MCPTool can be created without specifying 'tools'."""
    # Create MCPTool without specifying 'tools'
    mcp_tool = MCPTool(
        command="docker", args=["run", "-i", "--rm", "mcp/filesystem", "/projects"]
    )

    # Verify 'tools' is None by default
    assert mcp_tool.tools is None

    # Test with specified 'tools'
    mcp_tool_with_tools = MCPTool(
        command="docker",
        args=["run", "-i", "--rm", "mcp/filesystem", "/projects"],
        tools=["write_file", "read_file"],
    )

    # Verify 'tools' is properly set
    assert mcp_tool_with_tools.tools == ["write_file", "read_file"]


def test_agent_config_with_mcp_tool():
    """Test that AgentConfig can be created with MCPTool that has no 'tools' specified."""
    # Create MCPTool without 'tools'
    mcp_tool = MCPTool(
        command="docker", args=["run", "-i", "--rm", "mcp/filesystem", "/projects"]
    )

    # Create AgentConfig with the MCPTool
    agent_config = AgentConfig(model_id="gpt-4o", tools=[mcp_tool])

    # Verify the MCPTool was properly included
    assert len(agent_config.tools) == 1
    assert isinstance(agent_config.tools[0], MCPTool)
    assert agent_config.tools[0].tools is None
