from litellm import BaseModel
from pydantic import ConfigDict

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_output_type(agent_framework: AgentFramework) -> None:
    class TestOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        city_name: str

    agent = AnyAgent.create(
        agent_framework,
        AgentConfig(
            model_id="gpt-4.1-mini",
            output_type=TestOutput,
        ),
    )

    result = agent.run("What is the capital of France?")
    assert isinstance(result.final_output, TestOutput)
    assert result.final_output.city_name == "Paris"
