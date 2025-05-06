import os

import pandas as pd

from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.logging import logger
from any_agent.tracing.trace import AgentTrace

# Use the shared logger


def save_evaluation_results(
    evaluation_case: EvaluationCase,
    output_path: str,
    output_message: str,
    trace: AgentTrace,
    hypothesis_answer: str,
    passed_checks: int,
    failed_checks: int,
    score: float,
) -> None:
    """Save evaluation results to the specified output path.

    Args:
        evaluation_case: Path to the test case file
        agent_config: Path to the agent configuration file
        output_path: Path to save the results
        output_message: Formatted output message with evaluation details
        trace: The trace
        hypothesis_answer: The extracted hypothesis answer
        passed_checks: Number of passed checkpoints
        failed_checks: Number of failed checkpoints
        score: Evaluation score as a percentage

    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                        "config": evaluation_case.model_dump(),
                        "evaluation_case_path": evaluation_case.evaluation_case_path,
                        "output_message": output_message,
                        "trace": trace.model_dump_json(),
                        "hypothesis_answer": hypothesis_answer,
                        "passed_checks": passed_checks,
                        "failed_checks": failed_checks,
                        "score": round(score, 2),
                    },
                ],
            ),
        ],
    )
    logger.info(f"Writing output to {output_path}")
    df.to_json(output_path, orient="records", lines=True)
