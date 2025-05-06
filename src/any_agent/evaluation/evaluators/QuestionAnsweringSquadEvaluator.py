from collections.abc import Sequence

import evaluate.loading
from typing_extensions import TypedDict

from any_agent.evaluation.evaluation_case import GroundTruthAnswer
from any_agent.evaluation.evaluators.schemas import EvaluationResult


class AnswerDetails(TypedDict):
    answer_start: Sequence[int]
    text: Sequence[str]


class GroundTruthAnswers(TypedDict):
    id: str
    answers: AnswerDetails


class QuestionAnsweringSquadEvaluator:
    """Directly compares answers without using LLM-as-judge."""

    def __init__(self) -> None:
        self.metric = evaluate.loading.load("squad")

    def evaluate(
        self,
        hypothesis_answer: str,
        ground_truth_answer: Sequence[GroundTruthAnswer],
    ) -> list[EvaluationResult]:
        """Directly compare answers using simple matching."""
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
        result = self.metric.compute(
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
