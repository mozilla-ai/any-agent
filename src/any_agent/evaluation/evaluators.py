from collections.abc import Sequence
from typing import TYPE_CHECKING

from any_agent.evaluation.schemas import CheckpointCriteria, EvaluationResult
from any_agent.logging import logger
from any_agent.tracing.processors.base import TracingProcessor

if TYPE_CHECKING:
    from any_agent.tracing.trace import AgentTrace

import json
import re
from textwrap import dedent

import evaluate.loading
from litellm import completion

from any_agent.evaluation.schemas import GroundTruthAnswer, GroundTruthAnswers


def llm_evaluate_with_criterion(
    model: str,
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
        model=model,
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


def evaluate_checkpoint(
    model: str,
    trace: "AgentTrace",
    checkpoints: Sequence[CheckpointCriteria],
    processor: TracingProcessor,
) -> list[EvaluationResult]:
    """Verify each checkpoint against the trace data using LLM.

    Args:
        model: The model to use for evaluation
        trace: The trace data to evaluate
        checkpoints: List of checkpoint criteria to verify
        processor: Trace processor to extract evidence

    Returns:
        List of evaluation results

    """
    evidence = processor.extract_evidence(trace)
    evidence = evidence.replace("<", "\\<").replace(">", "\\>")
    logger.debug(f"""Evidence\n{evidence}\n""")
    results = []

    for checkpoint in checkpoints:
        evaluation = llm_evaluate_with_criterion(
            model,
            criteria=checkpoint.criteria,
            points=checkpoint.points,
            evidence=evidence,
        )
        results.append(evaluation)

    return results


def evaluate_hypothesis(
    model: str,
    hypothesis_final_output: str,
    ground_truth_answer_dict: Sequence[GroundTruthAnswer],
    ground_truth_checkpoints: Sequence[CheckpointCriteria],
) -> list[EvaluationResult]:
    """Verify if the final answer meets all specified criteria."""
    results = []

    for criterion in ground_truth_checkpoints:
        evaluation = llm_evaluate_with_criterion(
            model=model,
            criteria=criterion.criteria,
            points=criterion.points,
            ground_truth_output=ground_truth_answer_dict,
            hypothesis_final_output=hypothesis_final_output,
        )

        results.append(evaluation)

    return results


def evaluate_qa_squad(
    hypothesis_answer: str,
    ground_truth_answer: Sequence[GroundTruthAnswer],
) -> list[EvaluationResult]:
    """Directly compare answers using simple matching."""
    metric = evaluate.loading.load("squad")
    # format the answers so that they're dicts with 'id' and 'prediction' keys for hypo
    # and the ref has id and answers keys
    hypothesis_answers = [{"id": "1", "prediction_text": hypothesis_answer}]
    ground_truth_answers: list[GroundTruthAnswers] = [
        {
            "id": "1",
            "answers": {
                "answer_start": [0],
                "text": [str(ground_truth_answer[0]["value"])],
            },
        },
    ]
    # Use the SQuAD metric to compare answers
    result = metric.compute(
        predictions=hypothesis_answers,
        references=ground_truth_answers,
    )

    assert result, "The result of the evaluation is empty"

    match = EvaluationResult(
        passed=int(result["exact_match"]) == 1,
        reason=f"Partial Match (F1) score is {round(result['f1'], 2)}",
        criteria="Is the answer a direct match?",
        points=1,
    )
    return [match]
