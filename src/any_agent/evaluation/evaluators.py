from __future__ import annotations

from typing import TYPE_CHECKING

from any_agent.evaluation.schemas import CheckpointCriteria, EvaluationResult
from any_agent.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_agent.tracing.agent_trace import AgentTrace

import json
import re
from json import JSONDecodeError
from textwrap import dedent

from litellm import completion

from any_agent.evaluation.schemas import GroundTruthAnswer

MAX_EVIDENCE_LENGTH: int = 500


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
    evidence = f"Number of steps taken: {len(trace.spans)}\n\n"
    evidence = f"Number of tokens used: {trace.tokens.total_tokens}\n\n"

    for idx, span in enumerate(trace.spans):
        evidence += f"### Step {idx}\n"
        if idx == 0:
            input_val = span.attributes.get("gen_ai.input.messages")
            # messages should always be json
            if input_val:
                input_json = json.loads(input_val)
                evidence += f"Input: {json.dumps(input_json, indent=2)}\n\n"

        tool_args = span.attributes.get("gen_ai.tool.args")
        if tool_args:
            args_json = json.loads(tool_args)
            tool_name = span.attributes.get("gen_ai.tool.name")
            evidence += f"Tool called: {tool_name}\n\n"
            evidence += f"Tool arguments: {json.dumps(args_json, indent=2)}\n\n"

        output = span.attributes.get("gen_ai.output")
        if output:
            try:
                output_json = json.loads(output)
                # the output can be quite long, truncate if needed
                pretty_output = json.dumps(output_json, indent=2)
                pretty_output = (
                    pretty_output[:MAX_EVIDENCE_LENGTH] + "...[TRUNCATED]"
                    if len(pretty_output) > MAX_EVIDENCE_LENGTH
                    else pretty_output
                )
                evidence += f"Output: {pretty_output}\n\n"
            except JSONDecodeError:
                evidence += f"Output: {output}\n\n"
    evidence += f"Final Output: {trace.final_output}"
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


def _calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth strings."""
    # Normalize strings: lowercase and roughly split into words
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not truth_tokens:
        return 1.0 if not pred_tokens else 0.0

    if not pred_tokens:
        return 0.0

    # Calculate precision and recall
    common_tokens = pred_tokens.intersection(truth_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)


def evaluate_final_output(
    final_output: str,
    ground_truth_answer: GroundTruthAnswer,
) -> EvaluationResult:
    """Compare answers using simple string matching and F1 score."""
    ground_truth_text = str(ground_truth_answer["value"])

    # Check for exact match (case-insensitive)
    exact_match = final_output.strip().lower() == ground_truth_text.strip().lower()

    # Calculate F1 score for partial matching
    f1_score = _calculate_f1_score(final_output, ground_truth_text)

    return EvaluationResult(
        passed=exact_match,
        reason=f"Partial Match (F1) score is {round(f1_score, 2)}",
        criteria="Is the answer a direct match?",
        points=1,
    )
