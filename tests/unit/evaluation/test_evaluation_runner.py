from pathlib import Path
from unittest.mock import MagicMock, patch

from any_agent.config import AgentFramework
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluation_runner import EvaluationRunner
from any_agent.evaluation.evaluators.schemas import EvaluationResult
from any_agent.tracing.trace import AgentTrace


def test_runner_add_case(evaluation_case: EvaluationCase, tmp_path: Path) -> None:
    output_path = str(tmp_path / "test_output")
    runner = EvaluationRunner(output_path)

    # Add the test case
    runner.add_evaluation_case(evaluation_case)

    # Try to add the same case again
    runner.add_evaluation_case(evaluation_case)

    # Check that the case is still in the list
    assert len(runner._evaluation_cases) == 1, (
        "Evaluation case should only be added once."
    )

    # change the case and add again

    new_case = evaluation_case.model_copy(deep=True)
    new_case.ground_truth[0]["value"] = 2.0
    runner.add_evaluation_case(new_case)
    assert len(runner._evaluation_cases) == 2, (
        "Evaluation case should be added again with different values."
    )


def test_evaluation_runner_add_trace(agent_trace: AgentTrace, tmp_path: Path) -> None:
    output_path = str(tmp_path / "test_output")
    runner = EvaluationRunner(output_path)

    runner.add_trace(agent_trace, AgentFramework.OPENAI)
    assert len(runner._traces) == 1, "Trace should be added to the runner."

    runner.add_trace(agent_trace, AgentFramework.OPENAI)
    assert len(runner._traces) == 1, "Trace should not be added again."

    second_trace = agent_trace.model_copy(deep=True)
    second_trace.spans[0].name = "Different Span"
    runner.add_trace(second_trace, AgentFramework.OPENAI)
    assert len(runner._traces) == 2, "Second trace should be added to the runner."


def test_evaluation_runner_runs_all_cases(
    evaluation_case: EvaluationCase, agent_trace: AgentTrace, tmp_path: Path
) -> None:
    """This unit test is designed to ensure that the EvaluationRunner properly iterates over all the traces and eval cases"""

    ### Set up the Runner and add the traces and eval cases
    output_path = str(tmp_path / "test_output")
    runner = EvaluationRunner(output_path)

    second_case = evaluation_case.model_copy(deep=True)
    second_case.ground_truth[0]["value"] = 2.0

    runner.add_evaluation_case(evaluation_case)
    runner.add_evaluation_case(second_case)

    # Add multiple traces
    second_trace = agent_trace.model_copy(deep=True)
    second_trace.spans[0].name = "Different Span"

    runner.add_trace(agent_trace, AgentFramework.OPENAI)
    runner.add_trace(second_trace, AgentFramework.OPENAI)

    #### Set up the mocks for the evaluators so that we don't actually call LLMs.
    mock_checkpoint_evaluator1 = MagicMock()
    mock_checkpoint_evaluator2 = MagicMock()
    mock_hypothesis_evaluator1 = MagicMock()
    mock_hypothesis_evaluator2 = MagicMock()
    mock_qa_evaluator1 = MagicMock()
    mock_qa_evaluator2 = MagicMock()

    ### Every evaluator will return the same result
    eval_result = [
        EvaluationResult(
            criteria="test criteria", passed=True, reason="test passed", points=1
        )
    ]
    for evaluator in [
        mock_checkpoint_evaluator1,
        mock_checkpoint_evaluator2,
        mock_hypothesis_evaluator1,
        mock_hypothesis_evaluator2,
        mock_qa_evaluator1,
        mock_qa_evaluator2,
    ]:
        evaluator.evaluate.return_value = eval_result

    with (
        patch(
            "any_agent.evaluation.evaluation_runner.CheckpointEvaluator"
        ) as mock_checkpoint_eval_cls,
        patch(
            "any_agent.evaluation.evaluation_runner.HypothesisEvaluator"
        ) as mock_hypothesis_eval_cls,
        patch(
            "any_agent.evaluation.evaluation_runner.QuestionAnsweringSquadEvaluator"
        ) as mock_qa_eval_cls,
        patch(
            "any_agent.evaluation.evaluation_runner.save_evaluation_results"
        ) as mock_save_results,
        patch(
            "any_agent.evaluation.evaluation_runner.TracingProcessor.create"
        ) as mock_processor_create,
    ):
        # Evaluators are created for each evaluation case (but not for each trace)
        mock_checkpoint_eval_cls.side_effect = [
            mock_checkpoint_evaluator1,
            mock_checkpoint_evaluator2,
        ]
        mock_hypothesis_eval_cls.side_effect = [
            mock_hypothesis_evaluator1,
            mock_hypothesis_evaluator2,
        ]
        mock_qa_eval_cls.side_effect = [mock_qa_evaluator1, mock_qa_evaluator2]

        # Mock processor is used to extract the hypothesis answer
        mock_processor = MagicMock()
        mock_processor._extract_hypothesis_answer.return_value = (
            "Mock hypothesis answer"
        )
        mock_processor_create.return_value = mock_processor

        # Run evaluation
        runner.run()

        # Verify that each evaluator class was instantiated twice (once per evaluation case)
        assert mock_checkpoint_eval_cls.call_count == 2
        assert mock_hypothesis_eval_cls.call_count == 2
        assert mock_qa_eval_cls.call_count == 2

        # Verify that each evaluator instance's evaluate method was called twice (once per trace)
        assert mock_checkpoint_evaluator1.evaluate.call_count == 2
        assert mock_checkpoint_evaluator2.evaluate.call_count == 2
        assert mock_hypothesis_evaluator1.evaluate.call_count == 2
        assert mock_hypothesis_evaluator2.evaluate.call_count == 2
        assert mock_qa_evaluator1.evaluate.call_count == 2
        assert mock_qa_evaluator2.evaluate.call_count == 2

        # Verify that save_evaluation_results was called once per trace per case (2 cases * 2 traces = 4 times)
        assert mock_save_results.call_count == 4
