from typing import Any, Literal

from any_agent.config import AgentFramework, Tool
from any_agent.tools.any_tool.any_tool import AnyToolBase


class TinyAgentTool(AnyToolBase[Tool]):
    framework: Literal[AgentFramework.TINYAGENT] = AgentFramework.TINYAGENT

    @classmethod
    def _validate_tool_type(cls, tool: Any) -> Tool:
        # Agno lets you pass callables directly in as tools ❤️
        return tool
