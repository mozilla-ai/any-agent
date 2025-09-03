import json
import os

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tracing.attributes import GenAI
from any_agent.tools.composio import CallableProvider

from composio import Composio


def test_composio(agent_framework: AgentFramework) -> None:
    cpo = Composio(CallableProvider())
    tools = cpo.tools.get(
        user_id=os.environ["COMPOSIO_USER_ID"],
        search="repository issues",
        toolkits=["GITHUB"],
        limit=10,
    )
    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id="openai:gpt-4.1-mini",
            instructions="You summarize GitHub Issues",
            tools=tools,
        ),
    )
    agent_trace = agent.run(
        "Summary of open issues with label `callbacks`, no `assignee` in `mozilla-ai/any-agent`"
    )
    tool_execution = next(s for s in agent_trace.spans if s.is_tool_execution())
    assert tool_execution is not None
    assert tool_execution.name == "execute_tool GITHUB_LIST_REPOSITORY_ISSUES"
    assert tool_execution.attributes[GenAI.OUTPUT] is not None
