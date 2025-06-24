import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import search_tavily


def test_no_instrument(
    agent_framework: AgentFramework,
) -> None:
    model_id = "gpt-4.1-nano"
    env_check = validate_environment("model_id")
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else None
    )

    agent = AnyAgent.create(
        agent_framework,
        AgentConfig(
            instructions="Use the available tools search the web for answers.",
            model_args=model_args,
            model_id=model_id,
            tools=[search_tavily],
        ),
    )

    assert agent._instrumentor

    agent_trace = agent.run(
        "How is the weather in Salvaterra de Miño?", instrument=False
    )

    assert not any(
        span.is_llm_call() or span.is_tool_execution() for span in agent_trace.spans
    )
