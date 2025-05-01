from fire import Fire

from any_agent.evaluation import EvaluationRunner
from any_agent.logging import logger


def do_eval(
    test_case_paths: list[str],
    telemetry_paths: list[str],
) -> None:
    logger.info("Starting evaluation...")
    runner = EvaluationRunner()

    for test_case_path in test_case_paths:
        logger.info(f"Loading test case from {test_case_path}")
        runner.add_test_case(test_case_path)

    for telemetry_path in telemetry_paths:
        logger.info(f"Loading telemetry from {telemetry_path}")
        runner.add_telemetry(telemetry_path)

    logger.info("Running evaluation...")
    runner.run()
    logger.info("Evaluation completed.")


def main() -> None:
    Fire(do_eval)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
