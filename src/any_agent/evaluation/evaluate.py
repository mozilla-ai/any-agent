from pydantic import BaseModel, ConfigDict

from any_agent.config import AgentFramework
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from any_agent.evaluation.evaluators.schemas import EvaluationResult
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentTrace


class TraceEvaluationResult(BaseModel):
    """Represents the result of evaluating a trace."""

    model_config = ConfigDict(extra="forbid")

    trace: AgentTrace
    hypothesis_answer: str
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
    checkpoint_evaluator = CheckpointEvaluator(model=evaluation_case.llm_judge)
    hypothesis_evaluator = HypothesisEvaluator(model=evaluation_case.llm_judge)
    qa_evaluator = QuestionAnsweringSquadEvaluator()
    processor = TracingProcessor.create(agent_framework)
    if not processor:
        msg = f"Processor for {agent_framework} not available."
        raise ValueError(msg)
    hypothesis_answer = processor._extract_hypothesis_answer(trace=trace)
    checkpoint_results = checkpoint_evaluator.evaluate(
        trace=trace,
        checkpoints=evaluation_case.checkpoints,
        processor=processor,
    )
    hypothesis_answer_results = hypothesis_evaluator.evaluate(
        hypothesis_final_output=hypothesis_answer,
        ground_truth_answer_dict=evaluation_case.ground_truth,
        ground_truth_checkpoints=evaluation_case.final_output_criteria,
    )

    if evaluation_case.ground_truth:
        direct_results = qa_evaluator.evaluate(
            hypothesis_answer=hypothesis_answer,
            ground_truth_answer=evaluation_case.ground_truth,
        )
    else:
        direct_results = []

    return TraceEvaluationResult(
        trace=trace,
        hypothesis_answer=hypothesis_answer,
        checkpoint_results=checkpoint_results,
        hypothesis_answer_results=hypothesis_answer_results,
        direct_results=direct_results,
    )
