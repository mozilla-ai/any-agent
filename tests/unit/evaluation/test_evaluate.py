from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework
from any_agent.evaluation import EvaluationCase, TraceEvaluationResult, evaluate
from any_agent.evaluation.evaluators.schemas import (
    EvaluationResult,
)
from any_agent.tracing.trace import AgentTrace


def test_evaluate_runs_all_evaluators(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace
) -> None:
    """This unit test is designed to ensure that the EvaluationRunner properly iterates over all the traces and eval cases"""
    #### Set up the mocks for the evaluators so that we don't actually call LLMs.
    mock_checkpoint_evaluator = MagicMock()
    mock_hypothesis_evaluator = MagicMock()
    mock_qa_evaluator = MagicMock()

    ### Every evaluator will return the same result
    eval_result = [
        EvaluationResult(
            criteria="test criteria", passed=True, reason="test passed", points=1
        )
    ]
    for evaluator in [
        mock_checkpoint_evaluator,
        mock_hypothesis_evaluator,
        mock_qa_evaluator,
    ]:
        evaluator.evaluate.return_value = eval_result

    with (
        patch(
            "any_agent.evaluation.evaluate.CheckpointEvaluator"
        ) as mock_checkpoint_eval_cls,
        patch(
            "any_agent.evaluation.evaluate.HypothesisEvaluator"
        ) as mock_hypothesis_eval_cls,
        patch(
            "any_agent.evaluation.evaluate.QuestionAnsweringSquadEvaluator"
        ) as mock_qa_eval_cls,
        patch(
            "any_agent.evaluation.evaluate.TracingProcessor.create"
        ) as mock_processor_create,
    ):
        # Evaluators are created for each evaluation case (but not for each trace)
        mock_checkpoint_eval_cls.side_effect = [
            mock_checkpoint_evaluator,
        ]
        mock_hypothesis_eval_cls.side_effect = [mock_hypothesis_evaluator]
        mock_qa_eval_cls.side_effect = [mock_qa_evaluator]

        # Mock processor is used to extract the hypothesis answer
        mock_processor = MagicMock()
        mock_processor._extract_hypothesis_answer.return_value = (
            "Mock hypothesis answer"
        )
        mock_processor_create.return_value = mock_processor

        evaluate(
            evaluation_case=evaluation_case,
            trace=agent_trace,
            agent_framework=AgentFramework.OPENAI,
        )

        assert mock_checkpoint_eval_cls.call_count == 1
        assert mock_hypothesis_eval_cls.call_count == 1
        assert mock_qa_eval_cls.call_count == 1

        assert mock_checkpoint_evaluator.evaluate.call_count == 1
        assert mock_hypothesis_evaluator.evaluate.call_count == 1
        assert mock_qa_evaluator.evaluate.call_count == 1


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

    hypothesis_results = [
        EvaluationResult(
            criteria="Hypothesis 1", passed=True, reason="Passed", points=5
        ),
        EvaluationResult(
            criteria="Hypothesis 2", passed=False, reason="Failed", points=1
        ),
    ]

    direct_results = [
        EvaluationResult(criteria="Direct 1", passed=True, reason="Passed", points=3),
        EvaluationResult(criteria="Direct 2", passed=True, reason="Passed", points=4),
        EvaluationResult(criteria="Direct 3", passed=False, reason="Failed", points=2),
    ]

    # Create a TraceEvaluationResult instance
    evaluation_result = TraceEvaluationResult(
        trace=agent_trace,
        hypothesis_answer="Test hypothesis",
        checkpoint_results=checkpoint_results,
        hypothesis_answer_results=hypothesis_results,
        direct_results=direct_results,
    )

    expected_score = 14 / 20

    # Check that the score property returns the correct value
    assert evaluation_result.score == expected_score, (
        f"Expected score {expected_score}, got {evaluation_result.score}"
    )

    # Test case with no points (should raise ValueError)
    zero_point_result = TraceEvaluationResult(
        trace=agent_trace,
        hypothesis_answer="Test hypothesis",
        checkpoint_results=[],
        hypothesis_answer_results=[],
        direct_results=[],
    )
    with pytest.raises(ValueError, match="Total points is 0, cannot calculate score."):
        zero_point_result.score  # noqa: B018
