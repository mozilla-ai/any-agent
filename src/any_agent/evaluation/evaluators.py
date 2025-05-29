from __future__ import annotations

from typing import TYPE_CHECKING

from any_agent.evaluation.schemas import CheckpointCriteria, EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_agent import AnyAgent
    from any_agent.tracing.agent_trace import AgentTrace
from any_agent.evaluation.agent import get_agent
from any_agent.evaluation.schemas import AgentOutput, GroundTruthAnswer


def evaluate_checkpoints(
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
    results = []

    checking_agent: AnyAgent = get_agent(trace, model)

    for checkpoint in checkpoints:
        if callable(checkpoint.criteria):
            eval_output = checkpoint.criteria(trace)
        else:
            # Agent as a Judge
            evaluation = checking_agent.run(prompt=checkpoint.criteria)
            # strip out the ```json and ``` from the final output if they exist
            if not evaluation.final_output:
                msg = "The evaluation result is empty"
                raise ValueError(msg)
            final_output = evaluation.final_output.replace("```json", "").replace(
                "```", ""
            )
            eval_output = AgentOutput.model_validate_json(final_output)
        result = EvaluationResult(
            passed=eval_output.passed,
            reason=eval_output.reasoning,
            criteria=checkpoint.criteria,
            points=checkpoint.points,
        )
        results.append(result)
    checking_agent.exit()
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
