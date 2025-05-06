import json

from fire import Fire

from any_agent.config import AgentFramework
from any_agent.evaluation import EvaluationRunner
from any_agent.evaluation.evaluation_case import EvaluationCase
from any_agent.logging import logger
from any_agent.tracing.trace import AgentSpan, AgentTrace


def do_eval(
    evaluation_case_paths: list[str],
    trace_paths: list[str],
    agent_framework: AgentFramework,
    output_path: str = "output/results.json",
) -> None:
    logger.info("Starting evaluation...")
    runner = EvaluationRunner(output_path=output_path)

    for evaluation_case_path in evaluation_case_paths:
        logger.info("Loading test case from %s", evaluation_case_path)
        evaluation_case = EvaluationCase.from_yaml(evaluation_case_path)
        runner.add_evaluation_case(evaluation_case)

    for trace_path in trace_paths:
        logger.info("Loading tracing from %s", trace_path)
        with open(trace_path, encoding="utf-8") as f:
            spans = json.loads(f.read())
        spans = [AgentSpan.model_validate_json(span) for span in spans]
        trace = AgentTrace(spans=spans)
        # dump the trace to a file
        with open("tmp.tmp", "w", encoding="utf-8") as f:
            f.write(trace.model_dump_json(indent=2))
        runner.add_trace(trace, agent_framework)

    logger.info("Running evaluation...")
    runner.run()
    logger.info("Evaluation completed.")


def main() -> None:
    Fire(do_eval)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
