from __future__ import annotations

from typing import TYPE_CHECKING

from any_agent.evaluation.schemas import CheckpointCriteria, EvaluationResult
from any_agent.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_agent.tracing.agent_trace import AgentTrace

import json
import re
from textwrap import dedent

import evaluate.loading
from litellm import completion

from any_agent.evaluation.schemas import GroundTruthAnswer, GroundTruthAnswers

MAX_EVIDENCE_LENGTH: int = 400


def llm_evaluate_with_criterion(
    model: str,
    criteria: str,
    points: int,
    evidence: str | None = None,
) -> EvaluationResult:
    """Evaluate a single criterion using LLM."""
    prompt = dedent(f"""
    Evaluate if the following criterion was met {"based on the provided evidence" if evidence else "in the agent's answer"}.

    Criterion: {criteria}
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


def _construct_evidence(trace: AgentTrace) -> str:
    evidence = "## Agent Execution\n\n"

    for idx, span in enumerate(trace.spans):
        evidence += f"### Call {idx}\n"

        call = {
            k: (
                v[:MAX_EVIDENCE_LENGTH] + "..."
                if isinstance(v, str) and len(v) > MAX_EVIDENCE_LENGTH
                else v
            )
            for k, v in span.attributes.items()
        }

        # Use ensure_ascii=False to prevent escaping Unicode characters
        evidence += json.dumps(call, indent=2, ensure_ascii=False) + "\n\n"

    return evidence


def evaluate_checkpoint(
    model: str,
    trace: AgentTrace,
    checkpoints: Sequence[CheckpointCriteria],
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
    evidence = _construct_evidence(trace)
    evidence = evidence.replace("<", "\\<").replace(">", "\\>")
    logger.debug(f"""Evidence\n{evidence}\n""")
    results = []

    for checkpoint in checkpoints:
        evaluation = llm_evaluate_with_criterion(
            model=model,
            criteria=checkpoint.criteria,
            points=checkpoint.points,
            evidence=evidence,
        )
        results.append(evaluation)

    return results


def evaluate_qa_squad(
    final_output: str,
    ground_truth_answer: GroundTruthAnswer,
) -> EvaluationResult:
    """Directly compare answers using simple matching."""
    metric = evaluate.loading.load("squad")
    # format the answers so that they're dicts with 'id' and 'prediction' keys for hypo
    # and the ref has id and answers keys
    predictions = [{"id": "1", "prediction_text": final_output}]
    ground_truth_answers: list[GroundTruthAnswers] = [
        {
            "id": "1",
            "answers": {
                "answer_start": [0],
                "text": [str(ground_truth_answer["value"])],
            },
        },
    ]
    # Use the SQuAD metric to compare answers
    result = metric.compute(
        predictions=predictions,
        references=ground_truth_answers,
    )

    assert result, "The result of the evaluation is empty"

    return EvaluationResult(
        passed=int(result["exact_match"]) == 1,
        reason=f"Partial Match (F1) score is {round(result['f1'], 2)}",
        criteria="Is the answer a direct match?",
        points=1,
    )
