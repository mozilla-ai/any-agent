from datetime import datetime

from any_agent import AgentFramework, AgentTrace
from any_agent.evaluation.agent_judge import AgentJudge
from any_agent.evaluation.llm_judge import LlmJudge
from any_agent.evaluation.schemas import EvaluationOutput
from any_agent.testing.helpers import get_default_agent_model_args


def test_llm_judge(agent_trace: AgentTrace) -> None:
    llm_judge = LlmJudge(
        model_id="openai/gpt-4.1-nano",
        model_args={
            "temperature": 0.0,
        },  # Because it's an llm not agent, the default_model_args are not used
    )
    result1 = llm_judge.run(
        context=str(agent_trace.spans_to_messages()),
        question="Do the messages contain the year 2025?",
    )
    assert isinstance(result1, EvaluationOutput)
    assert result1.passed, (
        f"Expected agent to call write_file tool, but evaluation failed: {result1.reasoning}"
    )


def test_agent_judge(agent_trace: AgentTrace) -> None:
    agent_judge = AgentJudge(
        model_id="openai/gpt-4.1-mini",
        model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
    )

    def get_current_year() -> str:
        """Get the current year"""
        return str(datetime.now().year)

    eval_trace = agent_judge.run(
        trace=agent_trace,
        question="Did the agent write the year to a file? Grab the messages from the trace and check if the write_file tool was called.",
        additional_tools=[get_current_year],
    )
    result = eval_trace.final_output
    assert isinstance(result, EvaluationOutput)
    assert result.passed, (
        f"Expected agent to write current year to file, but evaluation failed: {result.reasoning}"
    )
