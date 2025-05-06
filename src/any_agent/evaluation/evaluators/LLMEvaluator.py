import json
import re
from abc import ABC
from collections.abc import Sequence
from textwrap import dedent

from litellm import completion

from any_agent.evaluation.evaluation_case import CheckpointCriteria, GroundTruthAnswer
from any_agent.evaluation.evaluators.schemas import EvaluationResult


class LLMEvaluator(ABC):
    """Base class for evaluators that use LLM-as-judge."""

    def __init__(self, model: str):
        self.model = model

    def llm_evaluate_with_criterion(
        self,
        criteria: str,
        points: int,
        ground_truth_output: Sequence[CheckpointCriteria]
        | Sequence[GroundTruthAnswer]
        | None = None,
        hypothesis_final_output: str | None = None,
        evidence: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single criterion using LLM."""
        prompt = dedent(f"""
        Evaluate if the following criterion was met {"based on the provided evidence" if evidence else "in the agent's answer"}.

        Criterion: {criteria}
        """)

        if ground_truth_output:
            prompt += dedent(f"""
            Expected output: {json.dumps(ground_truth_output)}
            """)
        if hypothesis_final_output:
            prompt += dedent(f"""
            Agent's answer: {hypothesis_final_output}
            """)

        if evidence:
            prompt += dedent(f"""
            Trace evidence:
            {evidence}
            """)

        prompt += f"""

        Based on the {"evidence" if evidence else "comparison between the expected output and the actual final answer"},
        was this criterion satisfied? Answer with:
        1. "passed": true or false
        2. "reason": Brief explanation for your decision
        """
        prompt += """
        Output valid JSON with these three fields only, in the format:
        ```json
        {
            "passed": true,
            "reason": "I have them"
        }
        ```
        """

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content

        try:
            # Extract JSON from the response - looks for patterns like ```json {...} ``` or just {...}
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})",
                content,
                re.DOTALL,
            )

            if json_match:
                # Use the first matching group that captured content
                json_str = next(group for group in json_match.groups() if group)
                evaluation = json.loads(json_str)
            else:
                # Fallback: try parsing the whole content as JSON
                evaluation = json.loads(content)

            evaluation["criteria"] = criteria
        except (json.JSONDecodeError, AttributeError, StopIteration) as e:
            evaluation = {
                "passed": False,
                "reason": f"Failed to evaluate due to parsing: {e!s} \n Response: {content}",
                "criteria": criteria,
            }
        evaluation["points"] = points
        return EvaluationResult.model_validate(evaluation)
