from typing import Dict, List, Any

from any_agent.evaluation.evaluators.LLMEvaluator import LLMEvaluator
from any_agent.evaluation.evaluators.schemas import EvaluationResult
from any_agent.telemetry import TelemetryProcessor
from any_agent.evaluation.test_case import CheckpointCriteria
from any_agent.logging import logger


class CheckpointEvaluator(LLMEvaluator):
    """Evaluates checkpoints against telemetry"""

    def evaluate(
        self,
        telemetry: List[Dict[str, Any]],
        checkpoints: List[CheckpointCriteria],
        processor: TelemetryProcessor,
    ) -> List[EvaluationResult]:
        """
        Verify each checkpoint against the telemetry data using LLM

        Args:
            telemetry: The telemetry data to evaluate
            checkpoints: List of checkpoint criteria to verify
            processor: Telemetry processor to extract evidence

        Returns:
            List of evaluation results
        """
        evidence = processor.extract_evidence(telemetry)
        evidence = evidence.replace("<", "\\<").replace(">", "\\>")
        logger.info(f"""<yellow>Evidence\n{evidence}</yellow>\n""")
        results = []

        for checkpoint in checkpoints:
            evaluation = self.llm_evaluate_with_criterion(
                criteria=checkpoint.criteria,
                points=checkpoint.points,
                evidence=evidence,
            )
            results.append(evaluation)

        return results
