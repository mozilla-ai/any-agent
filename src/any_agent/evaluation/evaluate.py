from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluators import (
    evaluate_checkpoints,
    evaluate_final_output,
)
from any_agent.evaluation.schemas import TraceEvaluationResult
from any_agent.logging import logger
from any_agent.tracing.agent_trace import AgentTrace


def evaluate(
    evaluation_case: EvaluationCase,
    trace: AgentTrace,
) -> TraceEvaluationResult:
    checkpoint_results = evaluate_checkpoints(
        model=evaluation_case.llm_judge,
        trace=trace,
        checkpoints=evaluation_case.checkpoints,
    )

    if evaluation_case.ground_truth and trace.final_output:
        ground_truth_result = evaluate_final_output(
            final_output=trace.final_output,
            ground_truth_answer=evaluation_case.ground_truth,
        )
    else:
        logger.warning(
            "No ground truth answer found in the evaluation case or no hypothesis answer in the trace. Skipping direct evaluation."
        )
        ground_truth_result = None

    return TraceEvaluationResult(
        trace=trace,
        checkpoint_results=checkpoint_results,
        ground_truth_result=ground_truth_result,
    )
