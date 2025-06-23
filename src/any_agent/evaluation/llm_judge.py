from typing import Any

from litellm import acompletion
from litellm.utils import supports_response_schema
from pydantic import BaseModel

from any_agent.config import AgentFramework
from any_agent.evaluation.schemas import AgentOutput
from any_agent.evaluation.tools import EvaluationTools
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.utils.asyncio_sync import run_async_in_sync

DEFAULT_PROMPT = """Please evaluate the following agent execution trace:

AGENT TRACE:
{trace_text}

EVALUATION QUESTION:
{question}"""


LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator that analyzes agent execution traces to answer specific questions about agent performance and behavior.

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


class LLMJudge:
    """An LLM that evaluates the correctness of another agent's trace."""

    def __init__(
        self,
        model_id: str,
        framework: AgentFramework = AgentFramework.TINYAGENT,
        output_type: type[BaseModel] = AgentOutput,
        model_args: dict[str, Any] | None = None,
    ):
        if model_args is None:
            model_args = {}
        self.model_id = model_id
        self.framework = framework
        self.model_args = model_args
        self.output_type = output_type
        # If LiteLLM detects that the model supports response_format, set it to the output_type automatically
        if supports_response_schema(model=self.model_id):
            self.model_args["response_format"] = self.output_type

    def run(
        self,
        trace: AgentTrace,
        question: str,
    ) -> BaseModel:
        """Initialize the LLMJudge with a trace and model.

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
        prompt: str = DEFAULT_PROMPT,
    ) -> BaseModel:
        """Run the LLM asynchronously.

        Args:
            trace: The agent trace to evaluate
            question: The question to ask the agent
            prompt: The prompt to use for the LLM
        Returns:
            The evaluation result

        """
        # Get formatted trace messages
        evaluation_tools = EvaluationTools(trace)
        trace_text = evaluation_tools.get_messages_from_trace()

        # Create the evaluation prompt
        prompt = prompt.format(trace_text=trace_text, question=question)

        # Make the LLM call
        try:
            response = await acompletion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **self.model_args,
            )

        except Exception as e:
            # Return a failed evaluation with error details
            return AgentOutput(
                passed=False, reasoning=f"Error during LLM evaluation: {e!s}"
            )

        if not isinstance(response.choices[0].message, self.output_type):
            msg = "LLM response is not an AgentOutput instance."
            raise ValueError(msg)

        return response.choices[0].message
