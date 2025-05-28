from unittest.mock import MagicMock, patch

import pytest

from any_agent.evaluation import EvaluationCase, TraceEvaluationResult, evaluate
from any_agent.evaluation.schemas import (
    EvaluationResult,
)
from any_agent.tracing.agent_trace import AgentTrace


def test_evaluate_runs_all_evaluators(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace
) -> None:
    """This unit test checks that all evaluators are called when evaluating a trace."""
    #### Set up the mocks for the evaluators so that we don't actually call LLMs.
    mock_checkpoint_evaluate = MagicMock()
    mock_qa_evaluate = MagicMock()

    ### Every evaluate will return the same result
    eval_result = [
        EvaluationResult(
            criteria="test criteria", passed=True, reason="test passed", points=1
        )
    ]

    mock_checkpoint_evaluate.return_value = eval_result
    mock_qa_evaluate.return_value = eval_result[0]

    with (
        patch(
            "any_agent.evaluation.evaluate.evaluate_checkpoints",
            mock_checkpoint_evaluate,
        ),
        patch("any_agent.evaluation.evaluate.evaluate_final_output", mock_qa_evaluate),
    ):
        evaluate(
            evaluation_case=evaluation_case,
            trace=agent_trace,
        )

        assert mock_checkpoint_evaluate.call_count == 1
        assert mock_qa_evaluate.call_count == 1


def test_evaluate_when_no_final_output(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace
) -> None:
    """This unit test checks that the hypothesis and qa evaluators are not called when there is no final output."""
    #### Set up the mocks for the evaluators so that we don't actually call LLMs.
    mock_checkpoint_evaluate = MagicMock()
    mock_hypothesis_evaluate = MagicMock()
    mock_qa_evaluate = MagicMock()

    agent_trace.final_output = None

    ### Every evaluate will return the same result
    eval_result = [
        EvaluationResult(
            criteria="test criteria", passed=True, reason="test passed", points=1
        )
    ]
    for evaluate_fn in [
        mock_checkpoint_evaluate,
        mock_hypothesis_evaluate,
        mock_qa_evaluate,
    ]:
        evaluate_fn.return_value = eval_result

    with (
        patch(
            "any_agent.evaluation.evaluate.evaluate_checkpoints",
            mock_checkpoint_evaluate,
        ),
        patch("any_agent.evaluation.evaluate.evaluate_final_output", mock_qa_evaluate),
    ):
        evaluate(
            evaluation_case=evaluation_case,
            trace=agent_trace,
        )

        assert mock_checkpoint_evaluate.call_count == 1
        assert mock_hypothesis_evaluate.call_count == 0
        assert mock_qa_evaluate.call_count == 0


def test_trace_evaluation_result_score_calculation(agent_trace: AgentTrace) -> None:
    """Test that the score property of TraceEvaluationResult correctly calculates the ratio of passed points to total points."""

    # Create evaluation results with different point values and pass status
    checkpoint_results = [
        EvaluationResult(
            criteria="Criterion 1", passed=True, reason="Passed", points=2
        ),
        EvaluationResult(
            criteria="Criterion 2", passed=False, reason="Failed", points=3
        ),
    ]

    ground_truth_result = EvaluationResult(
        criteria="Direct 1", passed=True, reason="Passed", points=3
    )

    # Create a TraceEvaluationResult instance
    evaluation_result = TraceEvaluationResult(
        trace=agent_trace,
        checkpoint_results=checkpoint_results,
        ground_truth_result=ground_truth_result,
    )

    expected_score = 5 / 8

    # Check that the score property returns the correct value
    assert evaluation_result.score == expected_score, (
        f"Expected score {expected_score}, got {evaluation_result.score}"
    )

    # Test case with no points (should raise ValueError)
    zero_point_result = TraceEvaluationResult(trace=agent_trace, checkpoint_results=[])
    with pytest.raises(ValueError, match="Total points is 0, cannot calculate score."):
        zero_point_result.score  # noqa: B018
