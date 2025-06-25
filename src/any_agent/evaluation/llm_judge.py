from typing import Any

from litellm import acompletion
from litellm.utils import supports_response_schema
from pydantic import BaseModel

from any_agent.config import AgentFramework
from any_agent.evaluation.schemas import EvaluationOutput
from any_agent.utils.asyncio_sync import run_async_in_sync

DEFAULT_PROMPT_TEMPLATE = """Please answer the evaluation question given the following contextual information:

CONTEXT:
{context}

EVALUATION QUESTION:
{question}"""


LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator that analyzes contextual information to answer specific questions about agent performance and behavior.

You will be provided with:
1. Contextual information of an agent's execution that may be relevant to the evaluation question
2. A specific evaluation question to answer

Your task is to carefully analyze the context and provide a judgment on whether the agent's performance meets the criteria specified in the question.

EVALUATION GUIDELINES:
- Be objective and thorough in your analysis
- If the question asks about specific actions, look for evidence of those actions in the context
- If unsure, err on the side of being more critical rather than lenient

Your output must match the following JSON schema:
{response_schema}"""


class LlmJudge:
    def __init__(
        self,
        model_id: str,
        framework: AgentFramework = AgentFramework.TINYAGENT,
        output_type: type[BaseModel] = EvaluationOutput,
        model_args: dict[str, Any] | None = None,
        system_prompt: str = LLM_JUDGE_SYSTEM_PROMPT,
    ):
        if model_args is None:
            model_args = {}
        self.model_id = model_id
        self.framework = framework
        self.model_args = model_args
        self.output_type = output_type
        self.system_prompt = system_prompt.format(
            response_schema=self.output_type.model_json_schema()
        )
        # If LiteLLM detects that the model supports response_format, set it to the output_type automatically
        if supports_response_schema(model=self.model_id):
            self.model_args["response_format"] = self.output_type

    def _create_prompt(self, context: str, question: str, prompt: str) -> str:
        if "{context}" not in prompt or "{question}" not in prompt:
            msg = "Prompt must contain the following placeholders: {context} and {question}"
            raise ValueError(msg)
        return prompt.format(
            context=context,
            question=question,
        )

    def run(
        self,
        context: str,
        question: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> BaseModel:
        """Run the judge synchronously.

        Args:
            context: Any relevant information that may be needed to answer the question
            question: The question to ask the agent
            prompt_template: The prompt to use for the LLM

        Returns:
            The evaluation result

        """
        return run_async_in_sync(self.run_async(context, question, prompt_template))

    async def run_async(
        self,
        context: str,
        question: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> BaseModel:
        """Run the LLM asynchronously.

        Args:
            context: Any relevant information that may be needed to answer the question
            question: The question to ask the agent
            prompt_template: The prompt to use for the LLM

        Returns:
            The evaluation result

        """
        prompt = self._create_prompt(context, question, prompt_template)

        # Make the LLM call
        response = await acompletion(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **self.model_args,
        )

        return self.output_type.model_validate_json(
            response.choices[0].message["content"]
        )
