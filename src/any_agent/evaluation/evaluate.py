import json
from textwrap import dedent
from typing import Any, Dict, List
from any_agent.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from any_agent.evaluation.results_saver import save_evaluation_results
from any_agent.evaluation.test_case import TestCase
from any_agent.logging import logger
from any_agent.telemetry.telemetry import TelemetryProcessor


def evaluate_telemetry(test_case: TestCase, telemetry_path: str) -> bool:
    with open(telemetry_path, "r") as f:
        telemetry: List[Dict[str, Any]] = json.loads(f.read())
    logger.info(f"Telemetry loaded from {telemetry_path}")

    agent_framework = TelemetryProcessor.determine_agent_framework(telemetry)

    processor = TelemetryProcessor.create(agent_framework)
    hypothesis_answer = processor.extract_hypothesis_answer(trace=telemetry)

    checkpoint_evaluator = CheckpointEvaluator(model=test_case.llm_judge)
    checkpoint_results = checkpoint_evaluator.evaluate(
        telemetry=telemetry,
        checkpoints=test_case.checkpoints,
        processor=processor,
    )

    hypothesis_evaluator = HypothesisEvaluator(model=test_case.llm_judge)
    hypothesis_answer_results = hypothesis_evaluator.evaluate(
        hypothesis_final_answer=hypothesis_answer,
        ground_truth_answer_dict=test_case.ground_truth,
        ground_truth_checkpoints=test_case.final_answer_criteria,
    )

    if test_case.ground_truth:
        direct_evaluator = QuestionAnsweringSquadEvaluator()
        direct_results = direct_evaluator.evaluate(
            hypothesis_answer=hypothesis_answer,
            ground_truth_answer=test_case.ground_truth,
        )
    else:
        direct_results = []

    verification_results = (
        checkpoint_results + hypothesis_answer_results + direct_results
    )

    output_message = ""
    output_message += (
        f"""<yellow>Hypothesis Final answer extracted: {hypothesis_answer}</yellow>\n"""
    )
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
                - {check.reason}</green>"""
            )
            output_message += message + "\n"
    if failed_checks:
        for check in failed_checks:
            message = dedent(
                f"""
                <red>Failed:
                - {check.criteria}
                - {check.reason}</red>"""
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
        raise ValueError("No points were defined in the test case")
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
