from typing import List
import evaluate
from any_agent.evaluation.evaluators.schemas import EvaluationResult


class QuestionAnsweringSquadEvaluator:
    """Directly compares answers without using LLM-as-judge"""

    def __init__(self):
        self.metric = evaluate.load("squad")

    def evaluate(
        self, hypothesis_answer: str, ground_truth_answer: list
    ) -> List[EvaluationResult]:
        """Directly compare answers using simple matching"""

        # format the answers so that they're dicts with 'id' and 'prediction' keys for hypo
        # and the ref has id and answers keys
        hypothesis_answer = [{"id": "1", "prediction_text": hypothesis_answer}]
        ground_truth_answer = [
            {
                "id": "1",
                "answers": {
                    "answer_start": [0],
                    "text": [str(ground_truth_answer[0]["value"])],
                },
            }
        ]
        # Use the SQuAD metric to compare answers
        result = self.metric.compute(
            predictions=hypothesis_answer, references=ground_truth_answer
        )

        match = EvaluationResult(
            passed=True if int(result["exact_match"]) == 1 else False,
            reason=f"Partial Match (F1) score is {round(result['f1'], 2)}",
            criteria="Is the answer a direct match?",
            points=1,
        )
        return [match]
