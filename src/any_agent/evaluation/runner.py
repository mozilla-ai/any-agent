import json
from textwrap import dedent

from any_agent.config import AgentFramework
from any_agent.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from any_agent.evaluation.evaluators.schemas import EvaluationResult
from any_agent.evaluation.results_saver import save_evaluation_results
from any_agent.evaluation.test_case import TestCase
from any_agent.logging import logger
from any_agent.tracing import TracingProcessor
from any_agent.tracing.trace import AgentSpan, AgentTrace


class EvaluationRunner:
    def __init__(self) -> None:
        self._test_cases: list[TestCase] = []
        self._telemetry_paths: list[tuple[str, AgentFramework]] = []
        self.checkpoint_evaluator: CheckpointEvaluator | None = None
        self.hypothesis_evaluator: HypothesisEvaluator | None = None
        self.qa_evaluator: QuestionAnsweringSquadEvaluator | None = None

    def _setup_evaluators(self, test_case: TestCase) -> None:
        self.checkpoint_evaluator = CheckpointEvaluator(model=test_case.llm_judge)
        self.hypothesis_evaluator = HypothesisEvaluator(model=test_case.llm_judge)
        self.qa_evaluator = QuestionAnsweringSquadEvaluator()

    def add_test_case(self, test_case_path: str) -> None:
        """Add test case file path to the evaluation runner."""
        test_case = TestCase.from_yaml(test_case_path)
        if test_case not in self._test_cases:
            self._test_cases.append(test_case)
        else:
            logger.warning("Test case %s already added.", test_case_path)

    def add_telemetry(
        self, telemetry_path: str, agent_framework: AgentFramework
    ) -> None:
        """Add telemetry file path to the evaluation runner."""
        if telemetry_path not in [t[1] for t in self._telemetry_paths]:
            self._telemetry_paths.append((telemetry_path, agent_framework))
        else:
            logger.warning("Telemetry %salready added.", telemetry_path)

    def _run_telemetry_eval(
        self, test_case: TestCase, telemetry_path: str, agent_framework: AgentFramework
    ) -> None:
        with open(telemetry_path, encoding="utf-8") as f:
            spans = json.loads(f.read())
        spans = [AgentSpan.model_validate_json(span) for span in spans]
        trace = AgentTrace(spans=spans)
        logger.info("Telemetry loaded from %s", telemetry_path)
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
            checkpoints=test_case.checkpoints,
            processor=processor,
        )
        if not self.hypothesis_evaluator:
            msg = "HypothesisEvaluator not initialized."
            raise ValueError(msg)
        hypothesis_answer_results = self.hypothesis_evaluator.evaluate(
            hypothesis_final_answer=hypothesis_answer,
            ground_truth_answer_dict=test_case.ground_truth,
            ground_truth_checkpoints=test_case.final_answer_criteria,
        )

        if test_case.ground_truth:
            if not self.qa_evaluator:
                msg = "QuestionAnsweringSquadEvaluator not initialized."
                raise ValueError(msg)
            direct_results = self.qa_evaluator.evaluate(
                hypothesis_answer=hypothesis_answer,
                ground_truth_answer=test_case.ground_truth,
            )
        else:
            direct_results = []
        self._compile_results(
            test_case=test_case,
            telemetry_path=telemetry_path,
            hypothesis_answer=hypothesis_answer,
            checkpoint_results=checkpoint_results,
            hypothesis_answer_results=hypothesis_answer_results,
            direct_results=direct_results,
        )

    def _compile_results(
        self,
        test_case: TestCase,
        telemetry_path: str,
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
            test_case=test_case,
            output_path=test_case.output_path,
            output_message=output_message,
            telemetry_path=telemetry_path,
            hypothesis_answer=hypothesis_answer,
            passed_checks=len(passed_checks),
            failed_checks=len(failed_checks),
            score=score,
        )

    def _run_test_case(self, test_case: TestCase) -> None:
        self._setup_evaluators(test_case)
        for telemetry_path, agent_framework in self._telemetry_paths:
            self._run_telemetry_eval(test_case, telemetry_path, agent_framework)

    def run(self) -> None:
        """Run the evaluation for all test cases."""
        for test_case in self._test_cases:
            self._run_test_case(test_case)
