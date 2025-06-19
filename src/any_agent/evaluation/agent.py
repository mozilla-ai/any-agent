from collections.abc import Callable
from typing import Any

from any_agent import AgentConfig, AnyAgent
from any_agent.config import AgentFramework
from any_agent.evaluation.schemas import AgentOutput
from any_agent.evaluation.tools import AgentEvaluationTools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.utils.asyncio_sync import run_async_in_sync

AGENT_INSTRUCTIONS = f"""You are a helpful assistant that will be used to evaluate the correctness of an agent trace. Given a specific question regarding the quality of the something about the agent, utilize the appropriate tools in order to gather the answer needed in order to accurately answer the question. If you are asked anything about what the agent did, you should strongly consider using the get_evidence_from_spans tool to get the evidence. However, if the question is about specific details of the agent's actions, you don't necessarily need to use the get_evidence_from_spans tool.

Answer with:
1. "passed": true or false
2. "reasoning": Brief explanation for your decision

Your final answer must be a valid JSON string that conforms to the following JSON schema:

{AgentOutput.model_json_schema()}
"""


class AgentAsJudge:
    """An agent that evaluates the correctness of another agent's trace."""

    def __init__(
        self, model: str, framework: AgentFramework = AgentFramework.TINYAGENT
    ):
        self.model = model
        self.framework = framework

    def run(
        self,
        trace: AgentTrace,
        question: str,
        additional_tools: list[Callable[[], Any]] = [],
    ) -> AgentOutput:
        """Initialize the AgentAsJudge with a trace and model.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent

        Returns:
            The evaluation result

        """
        return run_async_in_sync(
            self.run_async(trace, question, additional_tools)
        )

    async def run_async(
        self,
        trace: AgentTrace,
        question: str,
        additional_tools: list[Callable[[], Any]] = [],
    ) -> AgentOutput:
        """Run the agent asynchronously.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent

        Returns:
            The evaluation result

        """
        tooling = AgentEvaluationTools(trace)

        agent_config = AgentConfig(
            model_id=self.model,
            instructions=AGENT_INSTRUCTIONS,
            tools=tooling.get_all_tools() + additional_tools,
            output_type=AgentOutput,
        )

        agent = await AnyAgent.create_async(
            self.framework,
            agent_config=agent_config,
        )
        agent_trace = await agent.run_async(question)
        if not isinstance(agent_trace.final_output, AgentOutput):
            msg = "Agent output is not an AgentOutput instance."
            raise ValueError(msg)
        return agent_trace.final_output
