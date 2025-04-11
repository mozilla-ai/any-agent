import os
import pandas as pd

from any_agent.evaluation.test_case import TestCase
from any_agent.logging import logger

# Use the shared logger


def save_evaluation_results(
    test_case: TestCase,
    output_path: str,
    output_message: str,
    telemetry_path: str,
    hypothesis_answer: str,
    passed_checks: int,
    failed_checks: int,
    score: float,
) -> None:
    """
    Save evaluation results to the specified output path.

    Args:
        test_case: Path to the test case file
        agent_config: Path to the agent configuration file
        output_path: Path to save the results
        output_message: Formatted output message with evaluation details
        telemetry_path: Path to the telemetry file used
        hypothesis_answer: The extracted hypothesis answer
        passed_checks: Number of passed checkpoints
        failed_checks: Number of failed checkpoints
        score: Evaluation score as a percentage
    """
    # See if the output_path file exists
    if os.path.exists(output_path):
        logger.info(f"Reading existing output from {output_path}")
        df = pd.read_json(output_path, orient="records", lines=True)
    else:
        logger.info(f"Creating new output file at {output_path}")
        df = pd.DataFrame()

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "config": test_case.model_dump(),
                        "test_case_path": test_case.test_case_path,
                        "output_message": output_message,
                        "telemetry_path": telemetry_path,
                        "hypothesis_answer": hypothesis_answer,
                        "passed_checks": passed_checks,
                        "failed_checks": failed_checks,
                        "score": round(score, 2),
                    }
                ]
            ),
        ]
    )
    logger.info(f"Writing output to {output_path}")
    df.to_json(output_path, orient="records", lines=True)
