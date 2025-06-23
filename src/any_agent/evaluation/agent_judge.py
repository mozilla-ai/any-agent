from collections.abc import Callable
from typing import Any

from any_agent import AgentConfig, AnyAgent
from any_agent.config import AgentFramework
from any_agent.evaluation.schemas import AgentOutput
from any_agent.evaluation.tools import EvaluationTools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.utils.asyncio_sync import run_async_in_sync
from pydantic import BaseModel

AGENT_INSTRUCTIONS = f"""You are a helpful assistant that will be used to evaluate the correctness of an agent trace.
Given a specific question regarding the quality of something about the agent, \
utilize the appropriate tools in order to gather what you need to accurately answer the question. 

Answer with:
1. "passed": true or false
2. "reasoning": Brief explanation for your decision (2-3 sentences max)"""


class AgentJudge:
    """An agent that evaluates the correctness of another agent's trace."""

    def __init__(
        self,
        model_id: str,
        framework: AgentFramework = AgentFramework.TINYAGENT,
        output_type: type[BaseModel] = AgentOutput,
        model_args: dict[str, Any] | None = None,
    ):
        self.model_id = model_id
        self.framework = framework
        self.model_args = model_args
        self.output_type = output_type



    def run(
        self,
        trace: AgentTrace,
        question: str,
        additional_tools: list[Callable[[], Any]] = [],
    ) -> BaseModel:
        """Run the agent judge.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent
            additional_tools: Additional tools to use for the agent

        Returns:
            The evaluation result

        """
        return run_async_in_sync(self.run_async(trace, question, additional_tools))

    async def run_async(
        self,
        trace: AgentTrace,
        question: str,
        additional_tools: list[Callable[[], Any]] = [],
    ) -> BaseModel:
        """Run the agent judge asynchronously.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent
            additional_tools: Additional tools to use for the agent
        Returns:
            The evaluation result

        """
        tooling = EvaluationTools(trace)

        agent_config = AgentConfig(
            model_id=self.model_id,
            instructions=AGENT_INSTRUCTIONS,
            tools=tooling.get_all_tools() + additional_tools,
            output_type=self.output_type,
            model_args=self.model_args,
        )

        agent = await AnyAgent.create_async(
            self.framework,
            agent_config=agent_config,
        )
        agent_trace = await agent.run_async(question)
        if not isinstance(agent_trace.final_output, self.output_type):
            msg = f"Agent output is not an {self.output_type} instance."
            raise ValueError(msg)
        return agent_trace.final_output
