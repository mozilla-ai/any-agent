from textwrap import dedent

from any_agent.config import AgentFramework
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from any_agent.evaluation.evaluators.schemas import EvaluationResult
from any_agent.evaluation.results_saver import save_evaluation_results
from any_agent.logging import logger
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentTrace


class EvaluationRunner:
    def __init__(self, output_path: str = "output") -> None:
        self._evaluation_cases: list[EvaluationCase] = []
        self._traces: list[AgentTrace] = []
        self._trace_frameworks: list[AgentFramework] = []
        self.checkpoint_evaluator: CheckpointEvaluator | None = None
        self.hypothesis_evaluator: HypothesisEvaluator | None = None
        self.qa_evaluator: QuestionAnsweringSquadEvaluator | None = None
        self.output_path: str = output_path

    def _setup_evaluators(self, evaluation_case: EvaluationCase) -> None:
        self.checkpoint_evaluator = CheckpointEvaluator(model=evaluation_case.llm_judge)
        self.hypothesis_evaluator = HypothesisEvaluator(model=evaluation_case.llm_judge)
        self.qa_evaluator = QuestionAnsweringSquadEvaluator()

    def add_evaluation_case(self, evaluation_case: EvaluationCase) -> None:
        """Add test case file path to the evaluation runner."""
        if evaluation_case not in self._evaluation_cases:
            self._evaluation_cases.append(evaluation_case)
        else:
            logger.warning("Test case %s already added.", evaluation_case)

    def add_trace(self, trace: AgentTrace, agent_framework: AgentFramework) -> None:
        """Add trace file path to the evaluation runner."""
        if trace not in self._traces:
            self._traces.append(trace)
            self._trace_frameworks.append(agent_framework)
        else:
            logger.warning("trace %salready added.", trace)

    def _run_trace_eval(
        self,
        evaluation_case: EvaluationCase,
        trace: AgentTrace,
        agent_framework: AgentFramework,
    ) -> None:
        processor = TracingProcessor.create(agent_framework)
        if not processor:
            msg = f"Processor for {agent_framework} not available."
            raise ValueError(msg)
        hypothesis_answer = processor._extract_hypothesis_answer(trace=trace)
        if not self.checkpoint_evaluator:
            msg = "CheckpointEvaluator not initialized."
            raise ValueError(msg)
        checkpoint_results = self.checkpoint_evaluator.evaluate(
            trace=trace,
            checkpoints=evaluation_case.checkpoints,
            processor=processor,
        )
        if not self.hypothesis_evaluator:
            msg = "HypothesisEvaluator not initialized."
            raise ValueError(msg)
        hypothesis_answer_results = self.hypothesis_evaluator.evaluate(
            hypothesis_final_output=hypothesis_answer,
            ground_truth_answer_dict=evaluation_case.ground_truth,
            ground_truth_checkpoints=evaluation_case.final_output_criteria,
        )

        if evaluation_case.ground_truth:
            if not self.qa_evaluator:
                msg = "QuestionAnsweringSquadEvaluator not initialized."
                raise ValueError(msg)
            direct_results = self.qa_evaluator.evaluate(
                hypothesis_answer=hypothesis_answer,
                ground_truth_answer=evaluation_case.ground_truth,
            )
        else:
            direct_results = []
        self._compile_results(
            evaluation_case=evaluation_case,
            trace=trace,
            hypothesis_answer=hypothesis_answer,
            checkpoint_results=checkpoint_results,
            hypothesis_answer_results=hypothesis_answer_results,
            direct_results=direct_results,
        )

    def _compile_results(
        self,
        evaluation_case: EvaluationCase,
        trace: AgentTrace,
        hypothesis_answer: str,
        checkpoint_results: list[EvaluationResult],
        hypothesis_answer_results: list[EvaluationResult],
        direct_results: list[EvaluationResult],
    ) -> None:
        verification_results = (
            checkpoint_results + hypothesis_answer_results + direct_results
        )

        output_message = ""
        output_message += f"""<yellow>Hypothesis Final answer extracted: {hypothesis_answer}</yellow>\n"""
        failed_checks = [r for r in verification_results if not r.passed]
        passed_checks = [r for r in verification_results if r.passed]
        missed_points = sum([r.points for r in failed_checks])
        won_points = sum([r.points for r in passed_checks])
        if passed_checks:
            for check in passed_checks:
                message = dedent(
                    f"""
                    <green>Passed:
                    - {check.criteria}
                    - {check.reason}</green>""",
                )
                output_message += message + "\n"
        if failed_checks:
            for check in failed_checks:
                message = dedent(
                    f"""
                    <red>Failed:
                    - {check.criteria}
                    - {check.reason}</red>""",
                )
                output_message += message + "\n"
        else:
            output_message += "<green>All checkpoints passed!</green>\n"
        output_message += f"<green>Passed checkpoints: {len(passed_checks)}</green>\n"
        output_message += f"<red>Failed checkpoints: {len(failed_checks)}</red>\n"
        output_message += "<green>=====================================</green>\n"
        output_message += (
            f"<green>Score: {won_points}/{won_points + missed_points}</green>\n"
        )
        output_message += "<green>=====================================</green>\n"
        logger.info(output_message)

        if won_points + missed_points == 0:
            msg = "No points were defined in the test case"
            raise ValueError(msg)
        score = won_points / (won_points + missed_points) * 100

        # Save the evaluation results
        save_evaluation_results(
            evaluation_case=evaluation_case,
            output_path=self.output_path,
            output_message=output_message,
            trace=trace,
            hypothesis_answer=hypothesis_answer,
            passed_checks=len(passed_checks),
            failed_checks=len(failed_checks),
            score=score,
        )

    def _run_evaluation_case(self, evaluation_case: EvaluationCase) -> None:
        self._setup_evaluators(evaluation_case)
        for trace, agent_framework in zip(
            self._traces, self._trace_frameworks, strict=True
        ):
            self._run_trace_eval(evaluation_case, trace, agent_framework)

    def run(self) -> None:
        """Run the evaluation for all test cases."""
        for evaluation_case in self._evaluation_cases:
            self._run_evaluation_case(evaluation_case)
