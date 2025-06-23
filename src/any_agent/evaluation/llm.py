from collections.abc import Callable
from typing import Any
import json

from any_agent import AgentConfig, AnyAgent
from any_agent.config import AgentFramework
from any_agent.evaluation.schemas import AgentOutput
from any_agent.evaluation.tools import AgentEvaluationTools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.utils.asyncio_sync import run_async_in_sync

from litellm import completion, acompletion

LLM_JUDGE_INSTRUCTIONS = """You are an expert evaluator that analyzes agent execution traces to answer specific questions about agent performance and behavior.

You will be provided with:
1. A detailed trace of an agent's execution showing all the steps it took
2. A specific evaluation question to answer

Your task is to carefully analyze the trace and provide a judgment on whether the agent's performance meets the criteria specified in the question.

TRACE FORMAT:
The trace shows the conversation flow between the user, assistant (agent), and any tools used. Each message includes:
- Role: system, user, assistant 
- Content: The actual message or tool execution details

EVALUATION GUIDELINES:
- Be objective and thorough in your analysis
- Focus on what actually happened in the trace, not what should have happened
- Consider the agent's reasoning, tool usage, and final output
- If the question asks about specific actions, look for evidence of those actions in the trace
- If unsure, err on the side of being more critical rather than lenient

Answer with:
1. "passed": true or false
2. "reasoning": Brief explanation for your decision (2-3 sentences max)

Your response MUST be valid JSON matching this schema:
{
  "passed": boolean,
  "reasoning": "string"
}"""


class LLMAsJudge:
    """An LLM that evaluates the correctness of another agent's trace."""

    def __init__(
        self, model: str, framework: AgentFramework = AgentFramework.TINYAGENT
    ):
        self.model = model
        self.framework = framework

    def run(
        self,
        trace: AgentTrace,
        question: str,
    ) -> AgentOutput:
        """Initialize the LLMAsJudge with a trace and model.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent

        Returns:
            The evaluation result

        """
        return run_async_in_sync(self.run_async(trace, question))

    async def run_async(
        self,
        trace: AgentTrace,
        question: str,
    ) -> AgentOutput:
        """Run the LLM asynchronously.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent

        Returns:
            The evaluation result

        """
        # Get formatted trace messages
        trace_messages = trace.spans_to_messages()

        # Format the trace into a readable string
        trace_text = ""
        for i, msg in enumerate(trace_messages):
            trace_text += f"{i + 1}. {msg.role.capitalize()}: {msg.content}\n"

        # Create the evaluation prompt
        prompt = f"""Please evaluate the following agent execution trace:

AGENT TRACE:
{trace_text}

EVALUATION QUESTION:
{question}

Based on the trace above, does the agent's performance meet the criteria specified in the question?"""

        # Make the LLM call
        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                response_format=AgentOutput,
            )

            # Extract the response content
            if not response.choices or not response.choices[0].message:
                raise ValueError("No response from LLM")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty response from LLM")

            # Parse the JSON response
            parsed_response = json.loads(response_content)
            return AgentOutput(
                passed=parsed_response["passed"],
                reasoning=parsed_response["reasoning"],
            )

        except Exception as e:
            # Return a failed evaluation with error details
            return AgentOutput(
                passed=False, reasoning=f"Error during LLM evaluation: {e!s}"
            )
