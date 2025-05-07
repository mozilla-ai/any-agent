from any_agent.config import AgentFramework
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from any_agent.evaluation.evaluators.schemas import TraceEvaluationResult
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentTrace


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
