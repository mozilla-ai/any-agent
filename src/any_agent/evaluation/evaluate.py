from pydantic import BaseModel, ConfigDict

from any_agent.config import AgentFramework
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluators import (
    evaluate_checkpoint,
    evaluate_hypothesis,
    evaluate_qa_squad,
)
from any_agent.evaluation.schemas import EvaluationResult
from any_agent.logging import logger
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentTrace


class TraceEvaluationResult(BaseModel):
    """Represents the result of evaluating a trace."""

    model_config = ConfigDict(extra="forbid")

    trace: AgentTrace
    hypothesis_answer: str | None
    checkpoint_results: list[EvaluationResult]
    hypothesis_answer_results: list[EvaluationResult]
    direct_results: list[EvaluationResult]

    @property
    def score(self) -> float:
        """Calculate the score based on the evaluation results."""
        all_results = (
            self.checkpoint_results
            + self.hypothesis_answer_results
            + self.direct_results
        )
        total_points = sum([result.points for result in all_results])
        if total_points == 0:
            msg = "Total points is 0, cannot calculate score."
            raise ValueError(msg)
        passed_points = sum([result.points for result in all_results if result.passed])
        return float(passed_points / total_points)


def evaluate(
    evaluation_case: EvaluationCase,
    trace: AgentTrace,
    agent_framework: AgentFramework,
) -> TraceEvaluationResult:
    processor = TracingProcessor.create(agent_framework)
    if not processor:
        msg = f"Processor for {agent_framework} not available."
        raise ValueError(msg)
    hypothesis_answer = trace.final_output
    checkpoint_results = evaluate_checkpoint(
        model=evaluation_case.llm_judge,
        trace=trace,
        checkpoints=evaluation_case.checkpoints,
        processor=processor,
    )
    if hypothesis_answer is not None:
        hypothesis_answer_results = evaluate_hypothesis(
            model=evaluation_case.llm_judge,
            hypothesis_final_output=hypothesis_answer,
            ground_truth_answer_dict=evaluation_case.ground_truth,
            ground_truth_checkpoints=evaluation_case.final_output_criteria,
        )
    else:
        logger.warning(
            "No hypothesis answer found in the trace. Skipping hypothesis evaluation."
        )
        hypothesis_answer_results = []

    if evaluation_case.ground_truth and hypothesis_answer is not None:
        direct_results = evaluate_qa_squad(
            hypothesis_answer=hypothesis_answer,
            ground_truth_answer=evaluation_case.ground_truth,
        )
    else:
        logger.warning(
            "No ground truth answer found in the evaluation case or no hypothesis answer in the trace. Skipping direct evaluation."
        )
        direct_results = []

    return TraceEvaluationResult(
        trace=trace,
        hypothesis_answer=hypothesis_answer,
        checkpoint_results=checkpoint_results,
        hypothesis_answer_results=hypothesis_answer_results,
        direct_results=direct_results,
    )
